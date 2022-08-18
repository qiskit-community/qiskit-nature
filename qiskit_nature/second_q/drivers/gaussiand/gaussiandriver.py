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

""" Gaussian Driver """

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
from typing import Any, Optional, Union, TYPE_CHECKING

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.constants import PERIODIC_TABLE
from qiskit_nature.units import DistanceUnit
from qiskit_nature.exceptions import UnsupportMethodError
import qiskit_nature.optionals as _optionals
from qiskit_nature.settings import settings
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.qcschema_translator import qcschema_to_problem
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.properties import DipoleMoment, ElectronicDipoleMoment
from qiskit_nature.second_q.properties.bases import ElectronicBasis
from qiskit_nature.second_q.properties.integrals import OneBodyElectronicIntegrals

from .gaussian_utils import run_g16
from ..electronic_structure_driver import ElectronicStructureDriver, MethodType

if TYPE_CHECKING:
    from .gauopen.QCMatEl import MatEl

logger = logging.getLogger(__name__)


@_optionals.HAS_GAUSSIAN.require_in_instance
class GaussianDriver(ElectronicStructureDriver):
    """
    Qiskit Nature driver using the Gaussian™ 16 program.

    See http://gaussian.com/gaussian16/

    This driver uses the Gaussian open-source Gaussian 16 interfacing code in
    order to access integrals and other electronic structure information as
    computed by G16 for the given molecule. The job control file, as provided
    here for the molecular configuration, is augmented for our needs here such
    as to have it output a MatrixElement file.
    """

    def __init__(
        self,
        config: Union[str, list[str]] = "# rhf/sto-3g scf(conventional)\n\n"
        "h2 molecule\n\n0 1\nH   0.0  0.0    0.0\nH   0.0  0.0    0.735\n\n",
    ) -> None:
        """
        Args:
            config: A molecular configuration conforming to Gaussian™ 16 format.

        Raises:
            QiskitNatureError: Invalid Input
        """
        super().__init__()
        if not isinstance(config, str) and not isinstance(config, list):
            raise QiskitNatureError(f"Invalid config for Gaussian Driver '{config}'")

        if isinstance(config, list):
            config = "\n".join(config)

        self._config = config

        self._mel: "MatEl" | None = None

    @staticmethod
    @_optionals.HAS_GAUSSIAN.require_in_call
    def from_molecule(
        molecule: MoleculeInfo,
        basis: str = "sto-3g",
        method: MethodType = MethodType.RHF,
        driver_kwargs: Optional[dict[str, Any]] = None,
    ) -> "GaussianDriver":
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
        GaussianDriver.check_method_supported(method)
        basis = GaussianDriver.to_driver_basis(basis)

        if molecule.units == DistanceUnit.ANGSTROM:
            units = "Angstrom"
        elif molecule.units == DistanceUnit.BOHR:
            units = "Bohr"
        else:
            raise QiskitNatureError(f"Unknown unit '{molecule.units.value}'")
        cfg1 = f"# {method.value}/{basis} UNITS={units} scf(conventional)\n\n"
        name = "".join(molecule.symbols)
        geom = "\n".join(
            [
                name + " " + " ".join(map(str, coord))
                for (name, coord) in zip(molecule.symbols, molecule.coords)
            ]
        )
        cfg2 = f"{name} molecule\n\n"
        cfg3 = f"{molecule.charge} {molecule.multiplicity}\n{geom}\n\n"

        return GaussianDriver(cfg1 + cfg2 + cfg3)

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
        Checks that Gaussian supports this method.
        Args:
            method: Method type

        Raises:
            UnsupportMethodError: If method not supported.
        """
        if method not in [MethodType.RHF, MethodType.ROHF, MethodType.UHF]:
            raise UnsupportMethodError(f"Invalid Gaussian method {method.value}.")

    def run(self) -> ElectronicStructureProblem:
        cfg = self._config
        while not cfg.endswith("\n\n"):
            cfg += "\n"

        logger.debug(
            "User supplied configuration raw: '%s'",
            cfg.replace("\r", "\\r").replace("\n", "\\n"),
        )
        logger.debug("User supplied configuration\n%s", cfg)

        # To the Gaussian section of the input file passed here as section string
        # add line '# Symm=NoInt output=(matrix,i4labels,mo2el) tran=full'
        # NB: Line above needs to be added in right context, i.e after any lines
        #     beginning with % along with any others that start with #
        # append at end the name of the MatrixElement file to be written

        file, fname = tempfile.mkstemp(suffix=".mat")
        os.close(file)

        cfg = GaussianDriver._augment_config(fname, cfg)
        logger.debug("Augmented control information:\n%s", cfg)

        self._config = cfg

        run_g16(cfg)

        self._mel = GaussianDriver._parse_matrix_file(fname)
        try:
            os.remove(fname)
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to remove MatrixElement file %s", fname)

        return self.to_problem()

    @staticmethod
    def _augment_config(fname: str, cfg: str) -> str:
        """Adds the extra config we need to the input file"""
        cfgaug = ""
        with io.StringIO() as outf:
            with io.StringIO(cfg) as inf:
                # Add our Route line at the end of any existing ones
                line = ""
                added = False
                while not added:
                    line = inf.readline()
                    if not line:
                        break
                    if line.startswith("#"):
                        outf.write(line)
                        while not added:
                            line = inf.readline()
                            if not line:
                                raise QiskitNatureError("Unexpected end of Gaussian input")
                            if not line.strip():
                                outf.write(
                                    "# Window=Full Int=NoRaff Symm=(NoInt,None) "
                                    "output=(matrix,i4labels,mo2el) tran=full\n"
                                )
                                added = True
                            outf.write(line)
                    else:
                        outf.write(line)

                # Now add our filename after the title and molecule but
                # before any additional data. We located
                # the end of the # section by looking for a blank line after
                # the first #. Allows comment lines
                # to be inter-mixed with Route lines if that's ever done.
                # From here we need to see two sections
                # more, the title and molecule so we can add the filename.
                added = False
                section_count = 0
                blank = True
                while not added:
                    line = inf.readline()
                    if not line:
                        raise QiskitNatureError("Unexpected end of Gaussian input")
                    if not line.strip():
                        blank = True
                        if section_count == 2:
                            break
                    else:
                        if blank:
                            section_count += 1
                            blank = False
                    outf.write(line)

                outf.write(line)
                outf.write(fname)
                outf.write("\n\n")

                # Whatever is left in the original config we just append without further inspection
                while True:
                    line = inf.readline()
                    if not line:
                        break
                    outf.write(line)

                cfgaug = outf.getvalue()

        return cfgaug

    @staticmethod
    def _parse_matrix_file(fname: str) -> "MatEl":
        """
        get_driver_class is used here because the discovery routine will load all the gaussian
        binary dependencies, if not loaded already. It won't work without it.
        """
        try:
            # add gauopen to sys.path so that binaries can be loaded
            gauopen_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gauopen")
            if gauopen_directory not in sys.path:
                sys.path.insert(0, gauopen_directory)
            # pylint: disable=import-outside-toplevel
            from .gauopen.QCMatEl import MatEl
        except ImportError as mnfe:
            msg = (
                (
                    "qcmatrixio extension not found. "
                    "See Gaussian driver readme to build qcmatrixio.F using f2py"
                )
                if mnfe.name == "qcmatrixio"
                else str(mnfe)
            )

            logger.info(msg)
            raise QiskitNatureError(msg) from mnfe

        _mel = MatEl(file=fname)
        logger.debug("MatrixElement file:\n%s", _mel)

        return _mel

    @classmethod
    def _from_matrix_file(cls, fname: str) -> GaussianDriver:
        ret = GaussianDriver()
        ret._mel = GaussianDriver._parse_matrix_file(fname)
        return ret

    def to_qcschema(self) -> QCSchema:
        moc = GaussianDriver._get_matrix(self._mel, "ALPHA MO COEFFICIENTS")
        moc_b = GaussianDriver._get_matrix(self._mel, "BETA MO COEFFICIENTS")
        if np.array_equal(moc, moc_b):
            logger.debug("ALPHA and BETA MO COEFFS identical, keeping only ALPHA")
            moc_b = None

        hcore = GaussianDriver._get_matrix(self._mel, "CORE HAMILTONIAN ALPHA")
        logger.debug("CORE HAMILTONIAN ALPHA %s", hcore.shape)
        hcore_b = GaussianDriver._get_matrix(self._mel, "CORE HAMILTONIAN BETA")
        if np.array_equal(hcore, hcore_b):
            # From Gaussian interfacing documentation: "The two core Hamiltonians are identical
            # unless a Fermi contact perturbation has been applied."
            logger.debug("CORE HAMILTONIAN ALPHA and BETA identical, keeping only ALPHA")
            hcore_b = None
        logger.debug(
            "CORE HAMILTONIAN BETA %s",
            "- Not present" if hcore_b is None else hcore_b.shape,
        )

        mohij = np.dot(np.dot(moc.T, hcore), moc)
        mohij_b = None
        if moc_b is not None:
            mohij_b = np.dot(np.dot(moc_b.T, hcore_b if hcore_b is not None else hcore), moc_b)

        eri = GaussianDriver._get_matrix(self._mel, "REGULAR 2E INTEGRALS")
        logger.debug("REGULAR 2E INTEGRALS %s", eri.shape)
        useao2e = False
        if moc_b is None and self._mel.matlist.get("BB MO 2E INTEGRALS") is not None:
            # It seems that when using ROHF, where alpha and beta coeffs are
            # the same, that integrals
            # for BB and BA are included in the output, as well as just AA
            # that would have been expected
            # Using these fails to give the right answer (is ok for UHF).
            # So in this case we revert to
            # using 2 electron ints in atomic basis from the output and
            # converting them ourselves.
            useao2e = True
            logger.info(
                "Identical A and B coeffs but BB ints are present - using regular 2E ints instead"
            )
        if useao2e:
            # eri are 2-body in AO. We can convert to MO via the ElectronicBasisTransform but using
            # ints in MO already, as in the else here, is better
            einsum_ao_to_mo = "pqrs,pi,qj,rk,sl->ijkl"
            mohijkl = np.einsum(
                einsum_ao_to_mo,
                eri,
                moc,
                moc,
                moc,
                moc,
                optimize=settings.optimize_einsum,
            )
            mohijkl_ba = None
            mohijkl_bb = None
            if moc_b is not None:
                mohijkl_ba = np.einsum(
                    einsum_ao_to_mo,
                    eri,
                    moc_b,
                    moc_b,
                    moc,
                    moc,
                    optimize=settings.optimize_einsum,
                )
                mohijkl_bb = np.einsum(
                    einsum_ao_to_mo,
                    eri,
                    moc_b,
                    moc_b,
                    moc_b,
                    moc_b,
                    optimize=settings.optimize_einsum,
                )
        else:
            # These are in MO basis but by default will be reduced in size by frozen core default so
            # to use them we need to add Window=Full above when we augment the config
            mohijkl = GaussianDriver._get_matrix(self._mel, "AA MO 2E INTEGRALS")
            logger.debug("AA MO 2E INTEGRALS %s", mohijkl.shape)
            mohijkl_bb = GaussianDriver._get_matrix(self._mel, "BB MO 2E INTEGRALS")
            logger.debug(
                "BB MO 2E INTEGRALS %s",
                "- Not present" if mohijkl_bb is None else mohijkl_bb.shape,
            )
            mohijkl_ba = GaussianDriver._get_matrix(self._mel, "BA MO 2E INTEGRALS")
            logger.debug(
                "BA MO 2E INTEGRALS %s",
                "- Not present" if mohijkl_ba is None else mohijkl_ba.shape,
            )

        orbs_energy = GaussianDriver._get_matrix(self._mel, "ALPHA ORBITAL ENERGIES")
        logger.debug("ORBITAL ENERGIES %s", orbs_energy)
        orbs_energy_b = GaussianDriver._get_matrix(self._mel, "BETA ORBITAL ENERGIES")
        logger.debug("BETA ORBITAL ENERGIES %s", orbs_energy_b)

        return self._to_qcschema(
            hij=hcore,
            hij_b=hcore_b,
            eri=eri,
            hij_mo=mohij,
            hij_mo_b=mohij_b,
            eri_mo=mohijkl,
            eri_mo_ba=mohijkl_ba,
            eri_mo_bb=mohijkl_bb,
            e_nuc=self._mel.scalar("ENUCREP"),
            e_ref=self._mel.scalar("ETOTAL"),
            mo_coeff=moc,
            mo_coeff_b=moc_b,
            mo_energy=orbs_energy,
            mo_energy_b=orbs_energy_b,
            symbols=[PERIODIC_TABLE[atom] for atom in self._mel.ian],
            coords=self._mel.c,
            multiplicity=self._mel.multip,
            charge=self._mel.icharg,
            method="RHF",
            basis="sto-3g",
            creator="Gaussian",
            version=self._mel.gversion,
            routine=self._config,
            nbasis=self._mel.nbasis,
            nmo=moc.shape[0],
            nalpha=(self._mel.ne + self._mel.multip - 1) // 2,
            nbeta=(self._mel.ne - self._mel.multip + 1) // 2,
            keywords={},
        )

    def to_problem(
        self,
        *,
        include_dipole: bool = True,
    ) -> ElectronicStructureProblem:
        qcschema = self.to_qcschema()

        problem = qcschema_to_problem(qcschema)

        if include_dipole:
            # dipole moment
            dipints = GaussianDriver._get_matrix(self._mel, "DIPOLE INTEGRALS")
            dipints = np.einsum("ijk->kji", dipints)

            x_dip_ints = OneBodyElectronicIntegrals(ElectronicBasis.AO, (dipints[0], None))
            y_dip_ints = OneBodyElectronicIntegrals(ElectronicBasis.AO, (dipints[1], None))
            z_dip_ints = OneBodyElectronicIntegrals(ElectronicBasis.AO, (dipints[2], None))

            x_dipole = DipoleMoment(
                "x", [x_dip_ints, x_dip_ints.transform_basis(problem.basis_transform)]
            )
            y_dipole = DipoleMoment(
                "y", [y_dip_ints, y_dip_ints.transform_basis(problem.basis_transform)]
            )
            z_dipole = DipoleMoment(
                "z", [z_dip_ints, z_dip_ints.transform_basis(problem.basis_transform)]
            )

            coords = np.reshape(self._mel.c, (len(self._mel.ian), 3))
            nucl_dip = np.einsum("i,ix->x", self._mel.ian, coords)
            nucl_dip = np.round(nucl_dip, decimals=8)

            problem.properties.electronic_dipole_moment = ElectronicDipoleMoment(
                [x_dipole, y_dipole, z_dipole],
                nuclear_dipole_moment=nucl_dip,
                reverse_dipole_sign=True,
            )

        return problem

    @staticmethod
    def _get_matrix(mel, name) -> np.ndarray:
        """
        Gaussian dimens values may be negative which it itself handles in expand
        but convert to all positive for use in reshape. Note: Fortran index ordering.
        """
        m_x = mel.matlist.get(name)
        if m_x is None:
            return None
        dims = tuple(abs(i) for i in m_x.dimens)
        mat = np.reshape(m_x.expand(), dims, order="F")
        return mat
