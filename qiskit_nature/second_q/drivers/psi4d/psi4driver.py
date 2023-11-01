# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Psi4 Driver """

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from qiskit_nature import QiskitNatureError
from qiskit_nature.exceptions import UnsupportMethodError
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.qcschema_translator import qcschema_to_problem
import qiskit_nature.optionals as _optionals

from ..electronic_structure_driver import ElectronicStructureDriver, MethodType, _QCSchemaData

logger = logging.getLogger(__name__)


@_optionals.HAS_PSI4.require_in_instance
class Psi4Driver(ElectronicStructureDriver):
    """
    Qiskit Nature driver using the Psi4 program.
    See http://www.psicode.org/
    """

    def __init__(
        self,
        config: str
        | list[
            str
        ] = "molecule h2 {\n  0 1\n  H  0.0 0.0 0.0\n  H  0.0 0.0 0.735\n  no_com\n  no_reorient\n}\n\n"
        "set {\n  basis sto-3g\n  scf_type pk\n  reference rhf\n",
    ) -> None:
        """
        Args:
            config: A molecular configuration conforming to Psi4 format.
        Raises:
            QiskitNatureError: Psi4 Driver configuration should be a string or list of strings.
        """
        super().__init__()
        if not isinstance(config, str) and not isinstance(config, list):
            raise QiskitNatureError(
                f"Psi4 Driver configuration should be a string or list of strings:'{config}'."
            )

        if isinstance(config, list):
            config = "\n".join(config)

        self._config = config
        self._qcschemadata = _QCSchemaData()

    @staticmethod
    @_optionals.HAS_PSI4.require_in_call
    def from_molecule(
        molecule: MoleculeInfo,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        driver_kwargs: dict[str, Any] | None = None,
    ) -> "Psi4Driver":
        """Creates a driver from a molecule.

        Args:
            molecule: the molecular information.
            basis: the basis set.
            method: the SCF method type.
            driver_kwargs: keyword arguments to be passed to driver.

        Returns:
            The constructed driver instance.

        Raises:
            QiskitNatureError: when an unknown unit is encountered.
        """
        # Ignore kwargs parameter for this driver
        del driver_kwargs
        Psi4Driver.check_method_supported(method)
        basis = Psi4Driver.to_driver_basis(basis)

        if molecule.units == DistanceUnit.ANGSTROM:
            units = "ang"
        elif molecule.units == DistanceUnit.BOHR:
            units = "bohr"
        else:
            raise QiskitNatureError(f"Unknown unit '{molecule.units.value}'")

        name = "".join(molecule.symbols)
        geom = "\n".join(
            [
                name + " " + " ".join(map(str, coord))
                for name, coord in zip(molecule.symbols, molecule.coords)
            ]
        )
        cfg1 = f"molecule {name} {{\nunits {units}\n"
        cfg2 = f"{molecule.charge} {molecule.multiplicity}\n"
        cfg3 = f"{geom}\nno_com\nno_reorient\n}}\n\n"
        cfg4 = f"set {{\n basis {basis}\n scf_type pk\n reference {method.value}\n}}"
        return Psi4Driver(cfg1 + cfg2 + cfg3 + cfg4)

    @staticmethod
    def to_driver_basis(basis: str) -> str:
        """Converts basis to a driver acceptable basis.

        Args:
            basis: The basis set to be used.

        Returns:
            A driver acceptable basis.
        """
        if basis == "sto3g":
            return "sto-3g"
        return basis

    @staticmethod
    def check_method_supported(method: MethodType) -> None:
        """Checks that Psi4 supports this method.

        Args:
            method: the SCF method type.

        Raises:
            UnsupportMethodError: If the method is not supported.
        """
        if method not in [MethodType.RHF, MethodType.ROHF, MethodType.UHF]:
            raise UnsupportMethodError(f"Invalid Psi4 method {method.value}.")

    def run(self) -> ElectronicStructureProblem:
        cfg = self._config

        psi4d_directory = Path(__file__).resolve().parent
        template_file = psi4d_directory.joinpath("_template.txt")

        input_text = [cfg]

        file_fd, hdf5_file = tempfile.mkstemp(suffix=".hdf5")
        os.close(file_fd)
        input_text += [f'_FILE_PATH = "{Path(hdf5_file).as_posix()}"']

        with open(template_file, "r", encoding="utf8") as file:
            input_text += [line.strip("\n") for line in file.readlines()]

        file_fd, input_file = tempfile.mkstemp(suffix=".inp")
        os.close(file_fd)
        with open(input_file, "w", encoding="utf8") as stream:
            stream.write("\n".join(input_text))

        file_fd, output_file = tempfile.mkstemp(suffix=".out")
        os.close(file_fd)
        try:
            Psi4Driver._run_psi4(input_file, output_file)
            if logger.isEnabledFor(logging.DEBUG):
                with open(output_file, "r", encoding="utf8") as file:
                    logger.debug("Psi4 output file:\n%s", file.read())
        finally:
            run_directory = os.getcwd()
            for local_file in os.listdir(run_directory):
                if local_file.endswith(".clean"):
                    os.remove(run_directory + "/" + local_file)
            try:
                os.remove("timer.dat")
            except Exception:  # pylint: disable=broad-except
                pass

            try:
                os.remove(input_file)
            except Exception:  # pylint: disable=broad-except
                pass

            try:
                os.remove(output_file)
            except Exception:  # pylint: disable=broad-except
                pass

        self._qcschemadata = _QCSchemaData.from_hdf5(hdf5_file)
        try:
            os.remove(hdf5_file)
        except Exception:  # pylint: disable=broad-except
            pass

        return self.to_problem()

    def to_qcschema(self, *, include_dipole: bool = True) -> QCSchema:
        """Extracts all available information after the driver was run into a :class:`.QCSchema`
        object.

        Args:
            include_dipole: whether to include the custom dipole integrals in the QCSchema.

        Returns:
            A :class:`.QCSchema` storing all extracted system data computed by the driver.
        """
        return Psi4Driver._to_qcschema(self._qcschemadata, include_dipole=include_dipole)

    def to_problem(
        self,
        *,
        basis: ElectronicBasis = ElectronicBasis.MO,
        include_dipole: bool = True,
    ) -> ElectronicStructureProblem:
        return qcschema_to_problem(
            self.to_qcschema(include_dipole=include_dipole),
            basis=basis,
            include_dipole=include_dipole,
        )

    @staticmethod
    def _run_psi4(input_file, output_file):
        process = None
        try:
            with subprocess.Popen(
                [_optionals.PSI4, input_file, output_file],
                stdout=subprocess.PIPE,
                universal_newlines=True,
            ) as process:
                stdout, _ = process.communicate()
                process.wait()
        except Exception as ex:
            if process is not None:
                process.kill()

            raise QiskitNatureError(f"{_optionals.PSI4} run has failed") from ex

        if process.returncode != 0:
            errmsg = ""
            if stdout is not None:
                lines = stdout.splitlines()
                for line in lines:
                    logger.error(line)
                    errmsg += line + "\n"
            raise QiskitNatureError(
                f"{_optionals.PSI4} process return code {process.returncode}\n{errmsg}"
            )
