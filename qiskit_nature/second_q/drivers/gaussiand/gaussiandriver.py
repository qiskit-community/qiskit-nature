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
from typing import Any, Optional, Union

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.constants import BOHR, PERIODIC_TABLE
from qiskit_nature.exceptions import UnsupportMethodError
import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.properties.driver_metadata import DriverMetadata
from qiskit_nature.second_q.properties import (
    ElectronicStructureDriverResult,
    AngularMomentum,
    Magnetization,
    ParticleNumber,
    ElectronicEnergy,
    DipoleMoment,
    ElectronicDipoleMoment,
)
from qiskit_nature.second_q.properties.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.second_q.properties.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)

from .gaussian_utils import run_g16
from ..electronic_structure_driver import ElectronicStructureDriver, MethodType
from ..molecule import Molecule
from ..units_type import UnitsType


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

    @staticmethod
    @_optionals.HAS_GAUSSIAN.require_in_call
    def from_molecule(
        molecule: Molecule,
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

        if molecule.units == UnitsType.ANGSTROM:
            units = "Angstrom"
        elif molecule.units == UnitsType.BOHR:
            units = "Bohr"
        else:
            raise QiskitNatureError(f"Unknown unit '{molecule.units.value}'")
        cfg1 = f"# {method.value}/{basis} UNITS={units} scf(conventional)\n\n"
        name = "".join([name for (name, _) in molecule.geometry])
        geom = "\n".join(
            [name + " " + " ".join(map(str, coord)) for (name, coord) in molecule.geometry]
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

    def run(self) -> ElectronicStructureDriverResult:
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

        driver_result = self._parse_matrix_file(fname)
        try:
            os.remove(fname)
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to remove MatrixElement file %s", fname)

        # inject runtime config
        driver_metadata = driver_result.get_property("DriverMetadata")
        driver_metadata.config = cfg

        return driver_result

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
    def _parse_matrix_file(fname: str, useao2e: bool = False) -> ElectronicStructureDriverResult:
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

        mel = MatEl(file=fname)
        logger.debug("MatrixElement file:\n%s", mel)

        driver_result = ElectronicStructureDriverResult()

        # molecule
        coords = np.reshape(mel.c, (len(mel.ian), 3))
        geometry: list[tuple[str, list[float]]] = []
        for atom, xyz in zip(mel.ian, coords):
            geometry.append((PERIODIC_TABLE[atom], BOHR * xyz))

        driver_result.molecule = Molecule(
            geometry,
            multiplicity=mel.multip,
            charge=mel.icharg,
        )

        # driver metadata
        driver_result.add_property(DriverMetadata("GAUSSIAN", mel.gversion, ""))

        # basis transform
        moc = GaussianDriver._get_matrix(mel, "ALPHA MO COEFFICIENTS")
        moc_b = GaussianDriver._get_matrix(mel, "BETA MO COEFFICIENTS")
        if np.array_equal(moc, moc_b):
            logger.debug("ALPHA and BETA MO COEFFS identical, keeping only ALPHA")
            moc_b = None

        nmo = moc.shape[0]

        basis_transform = ElectronicBasisTransform(
            ElectronicBasis.AO, ElectronicBasis.MO, moc, moc_b
        )
        driver_result.add_property(basis_transform)

        # particle number
        num_alpha = (mel.ne + mel.multip - 1) // 2
        num_beta = (mel.ne - mel.multip + 1) // 2

        driver_result.add_property(
            ParticleNumber(num_spin_orbitals=nmo * 2, num_particles=(num_alpha, num_beta))
        )

        # electronic energy
        hcore = GaussianDriver._get_matrix(mel, "CORE HAMILTONIAN ALPHA")
        logger.debug("CORE HAMILTONIAN ALPHA %s", hcore.shape)
        hcore_b = GaussianDriver._get_matrix(mel, "CORE HAMILTONIAN BETA")
        if np.array_equal(hcore, hcore_b):
            # From Gaussian interfacing documentation: "The two core Hamiltonians are identical
            # unless a Fermi contact perturbation has been applied."
            logger.debug("CORE HAMILTONIAN ALPHA and BETA identical, keeping only ALPHA")
            hcore_b = None
        logger.debug(
            "CORE HAMILTONIAN BETA %s",
            "- Not present" if hcore_b is None else hcore_b.shape,
        )
        one_body_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (hcore, hcore_b))
        one_body_mo = one_body_ao.transform_basis(basis_transform)

        eri = GaussianDriver._get_matrix(mel, "REGULAR 2E INTEGRALS")
        logger.debug("REGULAR 2E INTEGRALS %s", eri.shape)
        if moc_b is None and mel.matlist.get("BB MO 2E INTEGRALS") is not None:
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
        two_body_ao = TwoBodyElectronicIntegrals(ElectronicBasis.AO, (eri, None, None, None))
        two_body_mo: TwoBodyElectronicIntegrals
        if useao2e:
            # eri are 2-body in AO. We can convert to MO via the ElectronicBasisTransform but using
            # ints in MO already, as in the else here, is better
            two_body_mo = two_body_ao.transform_basis(basis_transform)
        else:
            # These are in MO basis but by default will be reduced in size by frozen core default so
            # to use them we need to add Window=Full above when we augment the config
            mohijkl = GaussianDriver._get_matrix(mel, "AA MO 2E INTEGRALS")
            logger.debug("AA MO 2E INTEGRALS %s", mohijkl.shape)
            mohijkl_bb = GaussianDriver._get_matrix(mel, "BB MO 2E INTEGRALS")
            logger.debug(
                "BB MO 2E INTEGRALS %s",
                "- Not present" if mohijkl_bb is None else mohijkl_bb.shape,
            )
            mohijkl_ba = GaussianDriver._get_matrix(mel, "BA MO 2E INTEGRALS")
            logger.debug(
                "BA MO 2E INTEGRALS %s",
                "- Not present" if mohijkl_ba is None else mohijkl_ba.shape,
            )
            two_body_mo = TwoBodyElectronicIntegrals(
                ElectronicBasis.MO, (mohijkl, mohijkl_ba, mohijkl_bb, None)
            )

        electronic_energy = ElectronicEnergy(
            [one_body_ao, two_body_ao, one_body_mo, two_body_mo],
            nuclear_repulsion_energy=mel.scalar("ENUCREP"),
            reference_energy=mel.scalar("ETOTAL"),
        )

        kinetic = GaussianDriver._get_matrix(mel, "KINETIC ENERGY")
        logger.debug("KINETIC ENERGY %s", kinetic.shape)
        electronic_energy.kinetic = OneBodyElectronicIntegrals(ElectronicBasis.AO, (kinetic, None))

        overlap = GaussianDriver._get_matrix(mel, "OVERLAP")
        logger.debug("OVERLAP %s", overlap.shape)
        electronic_energy.overlap = OneBodyElectronicIntegrals(ElectronicBasis.AO, (overlap, None))

        orbs_energy = GaussianDriver._get_matrix(mel, "ALPHA ORBITAL ENERGIES")
        logger.debug("ORBITAL ENERGIES %s", overlap.shape)
        orbs_energy_b = GaussianDriver._get_matrix(mel, "BETA ORBITAL ENERGIES")
        logger.debug("BETA ORBITAL ENERGIES %s", overlap.shape)
        orbital_energies = (orbs_energy, orbs_energy_b) if moc_b is not None else orbs_energy
        electronic_energy.orbital_energies = np.asarray(orbital_energies)

        driver_result.add_property(electronic_energy)

        # dipole moment
        dipints = GaussianDriver._get_matrix(mel, "DIPOLE INTEGRALS")
        dipints = np.einsum("ijk->kji", dipints)

        x_dip_ints = OneBodyElectronicIntegrals(ElectronicBasis.AO, (dipints[0], None))
        y_dip_ints = OneBodyElectronicIntegrals(ElectronicBasis.AO, (dipints[1], None))
        z_dip_ints = OneBodyElectronicIntegrals(ElectronicBasis.AO, (dipints[2], None))

        x_dipole = DipoleMoment("x", [x_dip_ints, x_dip_ints.transform_basis(basis_transform)])
        y_dipole = DipoleMoment("y", [y_dip_ints, y_dip_ints.transform_basis(basis_transform)])
        z_dipole = DipoleMoment("z", [z_dip_ints, z_dip_ints.transform_basis(basis_transform)])

        nucl_dip = np.einsum("i,ix->x", mel.ian, coords)
        nucl_dip = np.round(nucl_dip, decimals=8)

        driver_result.add_property(
            ElectronicDipoleMoment(
                [x_dipole, y_dipole, z_dipole],
                nuclear_dipole_moment=nucl_dip,
                reverse_dipole_sign=True,
            )
        )

        # extra properties
        # TODO: once https://github.com/Qiskit/qiskit-nature/issues/312 is fixed we can stop adding
        # these properties by default.
        # if not settings.dict_aux_operators:
        driver_result.add_property(AngularMomentum(nmo * 2))
        driver_result.add_property(Magnetization(nmo * 2))

        return driver_result

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
