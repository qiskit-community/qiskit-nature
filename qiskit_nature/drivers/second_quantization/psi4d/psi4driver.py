# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" PSI4 Driver """

import logging
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Union, List, Optional, Any, Dict

from qiskit_nature import QiskitNatureError
from qiskit_nature.properties.second_quantization.electronic import ElectronicStructureDriverResult
import qiskit_nature.optionals as _optionals

from ...qmolecule import QMolecule
from ..electronic_structure_driver import ElectronicStructureDriver, MethodType
from ...molecule import Molecule
from ...units_type import UnitsType
from ....exceptions import UnsupportMethodError

logger = logging.getLogger(__name__)


@_optionals.HAS_PSI4.require_in_instance
class PSI4Driver(ElectronicStructureDriver):
    """
    Qiskit Nature driver using the PSI4 program.

    See http://www.psicode.org/
    """

    def __init__(
        self,
        config: Union[
            str, List[str]
        ] = "molecule h2 {\n  0 1\n  H  0.0 0.0 0.0\n  H  0.0 0.0 0.735\n}\n\n"
        "set {\n  basis sto-3g\n  scf_type pk\n  reference rhf\n",
    ) -> None:
        """
        Args:
            config: A molecular configuration conforming to PSI4 format.

        Raises:
            QiskitNatureError: Invalid Input
        """
        super().__init__()
        if not isinstance(config, str) and not isinstance(config, list):
            raise QiskitNatureError(f"Invalid config for PSI4 Driver '{config}'")

        if isinstance(config, list):
            config = "\n".join(config)

        self._config = config

    @staticmethod
    @_optionals.HAS_PSI4.require_in_call("PSI4Driver from_molecule")
    def from_molecule(
        molecule: Molecule,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        driver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "PSI4Driver":
        """
        Args:
            molecule: molecule
            basis: basis set
            method: Hartree-Fock Method type
            driver_kwargs: kwargs to be passed to driver
        Returns:
            driver

        Raises:
            QiskitNatureError: Unknown unit
        """
        # Ignore kwargs parameter for this driver
        del driver_kwargs
        PSI4Driver.check_method_supported(method)
        basis = PSI4Driver.to_driver_basis(basis)

        if molecule.units == UnitsType.ANGSTROM:
            units = "ang"
        elif molecule.units == UnitsType.BOHR:
            units = "bohr"
        else:
            raise QiskitNatureError(f"Unknown unit '{molecule.units.value}'")
        name = "".join([name for (name, _) in molecule.geometry])
        geom = "\n".join(
            [name + " " + " ".join(map(str, coord)) for (name, coord) in molecule.geometry]
        )
        cfg1 = f"molecule {name} {{\nunits {units}\n"
        cfg2 = f"{molecule.charge} {molecule.multiplicity}\n"
        cfg3 = f"{geom}\nno_com\nno_reorient\n}}\n\n"
        cfg4 = f"set {{\n basis {basis}\n scf_type pk\n reference {method.value}\n}}"
        return PSI4Driver(cfg1 + cfg2 + cfg3 + cfg4)

    @staticmethod
    def to_driver_basis(basis: str) -> str:
        """
        Converts basis to a driver acceptable basis
        Args:
            basis: The basis set to be used
        Returns:
            driver acceptable basis
        """
        if basis == "sto3g":
            return "sto-3g"
        return basis

    @staticmethod
    def check_method_supported(method: MethodType) -> None:
        """
        Checks that PSI4 supports this method.
        Args:
            method: Method type

        Raises:
            UnsupportMethodError: If method not supported.
        """
        if method not in [MethodType.RHF, MethodType.ROHF, MethodType.UHF]:
            raise UnsupportMethodError(f"Invalid PSI4 method {method.value}.")

    def run(self) -> ElectronicStructureDriverResult:
        cfg = self._config

        psi4d_directory = Path(__file__).resolve().parent
        template_file = psi4d_directory.joinpath("_template.txt")
        qiskit_nature_directory = psi4d_directory.parent.parent

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        molecule = QMolecule()
        warnings.filterwarnings("default", category=DeprecationWarning)

        input_text = [cfg]
        input_text += ["import sys"]
        syspath = (
            "['"
            + qiskit_nature_directory.as_posix()
            + "','"
            + "','".join(Path(p).as_posix() for p in sys.path)
            + "']"
        )

        input_text += ["sys.path = " + syspath + " + sys.path"]
        input_text += ["import warnings"]
        input_text += ["from qiskit_nature.drivers.qmolecule import QMolecule"]
        input_text += ["warnings.filterwarnings('ignore', category=DeprecationWarning)"]
        input_text += [f'_q_molecule = QMolecule("{Path(molecule.filename).as_posix()}")']
        input_text += ["warnings.filterwarnings('default', category=DeprecationWarning)"]

        with open(template_file, "r", encoding="utf8") as file:
            input_text += [line.strip("\n") for line in file.readlines()]

        file_fd, input_file = tempfile.mkstemp(suffix=".inp")
        os.close(file_fd)
        with open(input_file, "w", encoding="utf8") as stream:
            stream.write("\n".join(input_text))

        file_fd, output_file = tempfile.mkstemp(suffix=".out")
        os.close(file_fd)
        try:
            PSI4Driver._run_psi4(input_file, output_file)
            if logger.isEnabledFor(logging.DEBUG):
                with open(output_file, "r", encoding="utf8") as file:
                    logger.debug("PSI4 output file:\n%s", file.read())
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

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        _q_molecule = QMolecule(molecule.filename)
        warnings.filterwarnings("default", category=DeprecationWarning)
        _q_molecule.load()
        # remove internal file
        _q_molecule.remove_file()
        _q_molecule.origin_driver_name = "PSI4"
        _q_molecule.origin_driver_config = cfg
        return ElectronicStructureDriverResult.from_legacy_driver_result(_q_molecule)

    @staticmethod
    def _run_psi4(input_file, output_file):

        # Run psi4.
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
                for i, _ in enumerate(lines):
                    logger.error(lines[i])
                    errmsg += lines[i] + "\n"
            raise QiskitNatureError(
                f"{_optionals.PSI4} process return code {process.returncode}\n{errmsg}"
            )
