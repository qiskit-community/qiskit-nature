# This code is part of Qiskit.
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

""" Gaussian Driver """

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
from typing import Any, TYPE_CHECKING

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
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.utils import get_einsum

from .gaussian_utils import run_g16
from ..electronic_structure_driver import ElectronicStructureDriver, MethodType, _QCSchemaData

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
        config: str | list[str] = "# rhf/sto-3g scf(conventional)\n\n"
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
        *,
        basis: str = "sto-3g",
        method: MethodType = MethodType.RHF,
        driver_kwargs: dict[str, Any] | None = None,
    ) -> "GaussianDriver":
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
        """Checks that Gaussian supports this method.

        Args:
            method: the SCF method type.

        Raises:
            UnsupportMethodError: If the method is not supported.
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

    def to_qcschema(self, *, include_dipole: bool = True) -> QCSchema:
        return GaussianDriver._qcschema_from_matrix_file(self._mel, include_dipole=include_dipole)

    @staticmethod
    def _qcschema_from_matrix_file(mel: MatEl, *, include_dipole: bool = True) -> QCSchema:
        einsum_func, _ = get_einsum()
        data = _QCSchemaData()

        data.mo_coeff = GaussianDriver._get_matrix(mel, "ALPHA MO COEFFICIENTS")
        data.mo_coeff_b = GaussianDriver._get_matrix(mel, "BETA MO COEFFICIENTS")
        if np.array_equal(data.mo_coeff, data.mo_coeff_b):
            logger.debug("ALPHA and BETA MO COEFFS identical, keeping only ALPHA")
            data.mo_coeff_b = None

        data.hij = GaussianDriver._get_matrix(mel, "CORE HAMILTONIAN ALPHA")
        logger.debug("CORE HAMILTONIAN ALPHA %s", data.hij.shape)
        data.hij_b = GaussianDriver._get_matrix(mel, "CORE HAMILTONIAN BETA")
        if np.array_equal(data.hij, data.hij_b):
            # From Gaussian interfacing documentation: "The two core Hamiltonians are identical
            # unless a Fermi contact perturbation has been applied."
            logger.debug("CORE HAMILTONIAN ALPHA and BETA identical, keeping only ALPHA")
            data.hij_b = None
        logger.debug(
            "CORE HAMILTONIAN BETA %s",
            "- Not present" if data.hij_b is None else data.hij_b.shape,
        )

        data.hij_mo = np.dot(np.dot(data.mo_coeff.T, data.hij), data.mo_coeff)
        if data.mo_coeff_b is not None:
            data.hij_mo_b = np.dot(
                np.dot(data.mo_coeff_b.T, data.hij_b if data.hij_b is not None else data.hij),
                data.mo_coeff_b,
            )

        # TODO: add support for symmetry-reduced integrals
        # NOTE: supporting this will likely require changes to the _augment_config method where we
        # currently enforce Symm=NoInt. Support for symmetry-reduced integrals in the matrix file
        # produced by Gaussian will need to be investigated.
        data.eri = GaussianDriver._get_matrix(mel, "REGULAR 2E INTEGRALS")
        logger.debug("REGULAR 2E INTEGRALS %s", data.eri.shape)
        useao2e = False
        if data.mo_coeff_b is None and mel.matlist.get("BB MO 2E INTEGRALS") is not None:
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
            data.eri_mo = einsum_func(
                einsum_ao_to_mo,
                data.eri,
                data.mo_coeff,
                data.mo_coeff,
                data.mo_coeff,
                data.mo_coeff,
                optimize=settings.optimize_einsum,
            )
            if data.mo_coeff_b is not None:
                data.eri_mo_ba = einsum_func(
                    einsum_ao_to_mo,
                    data.eri,
                    data.mo_coeff_b,
                    data.mo_coeff_b,
                    data.mo_coeff,
                    data.mo_coeff,
                    optimize=settings.optimize_einsum,
                )
                data.eri_mo_bb = einsum_func(
                    einsum_ao_to_mo,
                    data.eri,
                    data.mo_coeff_b,
                    data.mo_coeff_b,
                    data.mo_coeff_b,
                    data.mo_coeff_b,
                    optimize=settings.optimize_einsum,
                )
        else:
            # These are in MO basis but by default will be reduced in size by frozen core default so
            # to use them we need to add Window=Full above when we augment the config
            data.eri_mo = GaussianDriver._get_matrix(mel, "AA MO 2E INTEGRALS")
            logger.debug("AA MO 2E INTEGRALS %s", data.eri_mo.shape)
            data.eri_mo_bb = GaussianDriver._get_matrix(mel, "BB MO 2E INTEGRALS")
            logger.debug(
                "BB MO 2E INTEGRALS %s",
                "- Not present" if data.eri_mo_bb is None else data.eri_mo_bb.shape,
            )
            data.eri_mo_ba = GaussianDriver._get_matrix(mel, "BA MO 2E INTEGRALS")
            logger.debug(
                "BA MO 2E INTEGRALS %s",
                "- Not present" if data.eri_mo_ba is None else data.eri_mo_ba.shape,
            )

        data.mo_energy = GaussianDriver._get_matrix(mel, "ALPHA ORBITAL ENERGIES")
        logger.debug("ORBITAL ENERGIES %s", data.mo_energy)
        data.mo_energy_b = GaussianDriver._get_matrix(mel, "BETA ORBITAL ENERGIES")
        logger.debug("BETA ORBITAL ENERGIES %s", data.mo_energy_b)

        data.e_nuc = mel.scalar("ENUCREP")
        data.e_ref = mel.scalar("ETOTAL")
        data.symbols = [PERIODIC_TABLE[atom] for atom in mel.ian]
        data.coords = mel.c
        data.multiplicity = mel.multip
        data.charge = mel.icharg
        data.method = "RHF"
        data.basis = "sto-3g"
        data.creator = "Gaussian"
        data.version = mel.gversion
        data.nbasis = mel.nbasis
        data.nmo = data.mo_coeff.shape[0]
        data.nalpha = (mel.ne + mel.multip - 1) // 2
        data.nbeta = (mel.ne - mel.multip + 1) // 2

        if include_dipole:
            # dipole moment
            dipints = GaussianDriver._get_matrix(mel, "DIPOLE INTEGRALS")
            dipints = einsum_func("ijk->kji", dipints)

            data.dip_x = dipints[0]
            data.dip_y = dipints[1]
            data.dip_z = dipints[2]
            data.dip_mo_x_a = np.dot(np.dot(data.mo_coeff.T, data.dip_x), data.mo_coeff)
            data.dip_mo_y_a = np.dot(np.dot(data.mo_coeff.T, data.dip_y), data.mo_coeff)
            data.dip_mo_z_a = np.dot(np.dot(data.mo_coeff.T, data.dip_z), data.mo_coeff)
            if data.mo_coeff_b is not None:
                data.dip_mo_x_b = np.dot(np.dot(data.mo_coeff_b.T, data.dip_x), data.mo_coeff_b)
                data.dip_mo_y_b = np.dot(np.dot(data.mo_coeff_b.T, data.dip_y), data.mo_coeff_b)
                data.dip_mo_z_b = np.dot(np.dot(data.mo_coeff_b.T, data.dip_z), data.mo_coeff_b)

            coords = np.reshape(mel.c, (len(mel.ian), 3))
            nucl_dip = einsum_func("i,ix->x", mel.ian, coords)
            nucl_dip = np.round(nucl_dip, decimals=8)
            ref_dip = GaussianDriver._get_matrix(mel, "ELECTRIC DIPOLE MOMENT")
            ref_dip = np.round(ref_dip, decimals=8)
            elec_dip = ref_dip - nucl_dip

            logger.info("HF Electronic dipole moment: %s", elec_dip)
            logger.info("Nuclear dipole moment: %s", nucl_dip)
            logger.info("Total dipole moment: %s", ref_dip)

            data.dip_nuc = nucl_dip
            data.dip_ref = ref_dip  # type: ignore[assignment]

        return GaussianDriver._to_qcschema(data, include_dipole=include_dipole)

    def to_problem(
        self,
        *,
        basis: ElectronicBasis = ElectronicBasis.MO,
        include_dipole: bool = True,
    ) -> ElectronicStructureProblem:
        return GaussianDriver._problem_from_matrix_file(
            self._mel, basis=basis, include_dipole=include_dipole
        )

    @staticmethod
    def _problem_from_matrix_file(
        mel: MatEl,
        *,
        basis: ElectronicBasis = ElectronicBasis.MO,
        include_dipole: bool = True,
    ) -> ElectronicStructureProblem:
        qcschema = GaussianDriver._qcschema_from_matrix_file(mel, include_dipole=include_dipole)

        problem = qcschema_to_problem(qcschema, basis=basis, include_dipole=include_dipole)

        if include_dipole and problem.properties.electronic_dipole_moment is not None:
            problem.properties.electronic_dipole_moment.reverse_dipole_sign = True

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
