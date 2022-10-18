# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" PSI4 Driver """

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

from qiskit_nature import QiskitNatureError
from qiskit_nature.exceptions import UnsupportMethodError
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.second_q.properties import ElectronicDipoleMoment
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.qcschema_translator import (
    qcschema_to_problem,
    get_ao_to_mo_from_qcschema,
)

import qiskit_nature.optionals as _optionals

from ._qmolecule import _QMolecule
from ..electronic_structure_driver import ElectronicStructureDriver, MethodType, _QCSchemaData

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
            str, list[str]
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
        self._qmolecule = _QMolecule()

    @staticmethod
    @_optionals.HAS_PSI4.require_in_call
    def from_molecule(
        molecule: MoleculeInfo,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        driver_kwargs: Optional[dict[str, Any]] = None,
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
                for (name, coord) in zip(molecule.symbols, molecule.coords)
            ]
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

    def run(self) -> ElectronicStructureProblem:
        cfg = self._config

        psi4d_directory = Path(__file__).resolve().parent
        template_file = psi4d_directory.joinpath("_template.txt")
        qiskit_nature_directory = psi4d_directory.parent.parent

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

        with open(template_file, "r", encoding="utf8") as file:
            input_text += [line.strip("\n") for line in file.readlines()]

        file_fd, hdf5_file = tempfile.mkstemp(suffix=".hdf5")
        os.close(file_fd)

        input_text += [f'_q_molecule.save("{Path(hdf5_file).as_posix()}")']

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

        self._qmolecule = _QMolecule(hdf5_file)
        self._qmolecule.load()

        try:
            os.remove(hdf5_file)
        except Exception:  # pylint: disable=broad-except
            pass

        return self.to_problem()

    def to_qcschema(self) -> QCSchema:
        return PSI4Driver._qcschema_from_qmolecule(self._qmolecule)

    @staticmethod
    def _qcschema_from_qmolecule(qmolecule: _QMolecule) -> QCSchema:
        data = _QCSchemaData()
        data.hij = qmolecule.hcore
        data.hij_b = qmolecule.hcore_b
        data.eri = qmolecule.eri
        data.hij_mo = qmolecule.mo_onee_ints
        data.hij_mo_b = qmolecule.mo_onee_ints_b
        data.eri_mo = qmolecule.mo_eri_ints
        data.eri_mo_ba = qmolecule.mo_eri_ints_ba
        data.eri_mo_bb = qmolecule.mo_eri_ints_bb
        data.e_nuc = qmolecule.nuclear_repulsion_energy
        data.e_ref = qmolecule.hf_energy
        data.mo_coeff = qmolecule.mo_coeff
        data.mo_coeff_b = qmolecule.mo_coeff_b
        data.mo_energy = qmolecule.orbital_energies
        data.mo_energy_b = qmolecule.orbital_energies_b
        data.mo_occ = None
        data.mo_occ_b = None
        data.symbols = qmolecule.atom_symbol
        data.coords = qmolecule.atom_xyz.flatten()
        data.multiplicity = qmolecule.multiplicity
        data.charge = qmolecule.molecular_charge
        data.masses = qmolecule.masses
        data.method = None
        data.basis = None
        data.creator = "PSI4"
        data.version = qmolecule.origin_driver_version
        data.routine = None
        data.nbasis = None
        data.nmo = qmolecule.num_molecular_orbitals
        data.nalpha = qmolecule.num_alpha
        data.nbeta = qmolecule.num_beta
        data.keywords = None

        return PSI4Driver._to_qcschema(data)

    def to_problem(
        self,
        *,
        basis: ElectronicBasis = ElectronicBasis.MO,
        include_dipole: bool = True,
    ) -> ElectronicStructureProblem:
        qcschema = PSI4Driver._qcschema_from_qmolecule(self._qmolecule)

        problem = qcschema_to_problem(qcschema, basis=basis)

        if include_dipole:
            x_dip = ElectronicIntegrals.from_raw_integrals(self._qmolecule.x_dip_ints)
            y_dip = ElectronicIntegrals.from_raw_integrals(self._qmolecule.y_dip_ints)
            z_dip = ElectronicIntegrals.from_raw_integrals(self._qmolecule.z_dip_ints)

            if basis == ElectronicBasis.MO:
                basis_transform = get_ao_to_mo_from_qcschema(qcschema)

                x_dip = basis_transform.transform_electronic_integrals(x_dip)
                y_dip = basis_transform.transform_electronic_integrals(y_dip)
                z_dip = basis_transform.transform_electronic_integrals(z_dip)

            dipole_moment = ElectronicDipoleMoment(x_dip, y_dip, z_dip)
            dipole_moment.nuclear_dipole_moment = self._qmolecule.nuclear_dipole_moment
            dipole_moment.reverse_dipole_sign = self._qmolecule.reverse_dipole_sign

            problem.properties.electronic_dipole_moment = dipole_moment

        return problem

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
                for line in lines:
                    logger.error(line)
                    errmsg += line + "\n"
            raise QiskitNatureError(
                f"{_optionals.PSI4} process return code {process.returncode}\n{errmsg}"
            )
