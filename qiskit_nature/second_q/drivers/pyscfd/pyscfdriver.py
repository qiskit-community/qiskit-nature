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

"""The PySCF Driver."""

from __future__ import annotations

import inspect
import logging
import os
import tempfile
import warnings
from enum import Enum
from typing import Any

import numpy as np
from qiskit.utils.validation import validate_min

from qiskit_nature.units import DistanceUnit
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.qcschema_translator import qcschema_to_problem
from qiskit_nature.second_q.operators.symmetric_two_body import fold
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.settings import settings
import qiskit_nature.optionals as _optionals
from qiskit_nature.utils import get_einsum

from ..electronic_structure_driver import ElectronicStructureDriver, MethodType, _QCSchemaData

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyscf")


class InitialGuess(Enum):
    """Initial Guess Enum"""

    MINAO = "minao"
    HCORE = "1e"
    ONE_E = "1e"
    ATOM = "atom"


@_optionals.HAS_PYSCF.require_in_instance
class PySCFDriver(ElectronicStructureDriver):
    """A Second-Quantization driver for Qiskit Nature using the PySCF library.

    References:
        https://pyscf.org/
    """

    def __init__(
        self,
        atom: str | list[str] = "H 0.0 0.0 0.0; H 0.0 0.0 0.735",
        *,
        unit: DistanceUnit = DistanceUnit.ANGSTROM,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        xc_functional: str = "lda,vwn",
        xcf_library: str = "libxc",
        conv_tol: float = 1e-9,
        max_cycle: int = 50,
        init_guess: InitialGuess = InitialGuess.MINAO,
        max_memory: int | None = None,
        chkfile: str | None = None,
    ) -> None:
        """
        Args:
            atom: A string (or a list thereof) denoting the elements and coordinates of all atoms in
                the system. Two formats are allowed; first, the PySCF-style `XYZ` format which is a
                list of strings formatted as `{element symbol} {x_coord} {y_coord} {z_coord}`. If a
                single string is given, the list entries should be joined by `;` as in the example:
                `H 0.0 0.0 0.0; H 0.0 0.0 0.735`.
                Second, the `Z-Matrix` format which is explained at 1_. The previous example
                would be written as `H; H 3 0.735`.
                See also 2_ for more details on geometry specifications supported by PySCF.
            unit: Denotes the unit of coordinates. Valid values are given by the ``UnitsType`` enum.
            charge: The charge of the molecule.
            spin: The spin of the molecule. In accordance with PySCF's definition, the spin equals
                :math:`2*S`, where :math:`S` is the total spin number of the molecule.
            basis: A basis set name as recognized by PySCF (3_), e.g. `sto3g` (the default), `321g`,
                etc. Note, that more advanced configuration options like a Dictionary or custom
                basis sets are not allowed for the moment. Refer to 4_ for an extensive list of
                PySCF's valid basis set names.
            method: The SCF method type to be used for the PySCF calculation. While the name
                refers to HF methods, the PySCFDriver also supports KS methods. Refer to the
                ``MethodType`` for a list of the supported methods.
            xc_functional: One of the predefined Exchange-Correlation functional names as recognized
                by PySCF (5_). Defaults to PySCF's default: 'lda,vwn'. __Note: this setting only has
                an effect when a KS method is chosen for `method`.__
            xcf_library: The Exchange-Correlation functional library to be used. This can be either
                'libxc' (the default) or 'xcfun'. Depending on this value, a different set of values
                for `xc_functional` will be available. Refer to 5_ for more details.
            conv_tol: The SCF convergence tolerance. See 6_ for more details.
            max_cycle: The maximum number of SCF iterations. See 6_ for more details.
            init_guess: The method to make the initial guess for the SCF starting point. Valid
                values are given by the ``InitialGuess`` enum. See 6_ for more details.
            max_memory: The maximum memory that PySCF should use. See 6_ for more details.
            chkfile: The path to a PySCF checkpoint file from which to load a previously run
                calculation. The data stored in this file is assumed to be already converged.
                Refer to 6_ and 7_ for more details.

        Raises:
            QiskitNatureError: An invalid input was supplied.

        .. _1: https://en.wikipedia.org/wiki/Z-matrix_(chemistry)
        .. _2: https://pyscf.org/user/gto.html#geometry
        .. _3: https://pyscf.org/user/gto.html#basis-set
        .. _4: https://pyscf.org/pyscf_api_docs/pyscf.gto.basis.html#module-pyscf.gto.basis
        .. _5: https://pyscf.org/user/dft.html#predefined-xc-functionals-and-functional-aliases
        .. _6: https://pyscf.org/pyscf_api_docs/pyscf.scf.html#module-pyscf.scf.hf
        .. _7: https://pyscf.org/pyscf_api_docs/pyscf.lib.html#module-pyscf.lib.chkfile
        """
        super().__init__()
        # pylint: disable=import-error
        from pyscf import gto, scf

        # First, ensure that PySCF supports the method
        PySCFDriver.check_method_supported(method)

        if isinstance(atom, list):
            atom = ";".join(atom)
        elif isinstance(atom, str):
            atom = atom.replace("\n", ";")
        else:
            raise QiskitNatureError(
                f"`atom` must be either a `str` or `list[str]`, but you passed {atom}"
            )

        validate_min("max_cycle", max_cycle, 1)

        # we use the property-setter to deal with conversion
        self.atom = atom
        self._unit = unit
        self._charge = charge
        self._spin = spin
        self._basis = basis
        self._method = method
        self._xc_functional = xc_functional
        self.xcf_library = xcf_library  # validate choice in property setter
        self._conv_tol = conv_tol
        self._max_cycle = max_cycle
        self._init_guess = init_guess.value
        self._max_memory = max_memory
        self._chkfile = chkfile

        self._mol: gto.Mole = None
        self._calc: scf.HF = None

    @property
    def atom(self) -> str:
        """Returns the atom."""
        return self._atom

    @atom.setter
    def atom(self, atom: str | list[str]) -> None:
        """Sets the atom."""
        if isinstance(atom, list):
            atom = ";".join(atom)
        self._atom = atom.replace("\n", ";")

    @property
    def unit(self) -> DistanceUnit:
        """Returns the unit."""
        return self._unit

    @unit.setter
    def unit(self, unit: DistanceUnit) -> None:
        """Sets the unit."""
        self._unit = unit

    @property
    def charge(self) -> int:
        """Returns the charge."""
        return self._charge

    @charge.setter
    def charge(self, charge: int) -> None:
        """Sets the charge."""
        self._charge = charge

    @property
    def spin(self) -> int:
        """Returns the spin."""
        return self._spin

    @spin.setter
    def spin(self, spin: int) -> None:
        """Sets the spin."""
        self._spin = spin

    @property
    def basis(self) -> str:
        """return basis"""
        return self._basis

    @basis.setter
    def basis(self, value: str) -> None:
        """set basis"""
        self._basis = value

    @property
    def method(self) -> MethodType:
        """Returns Hartree-Fock/Kohn-Sham method"""
        return self._method

    @method.setter
    def method(self, value: MethodType) -> None:
        """Sets Hartree-Fock/Kohn-Sham method"""
        self._method = value

    @property
    def xc_functional(self) -> str:
        """Returns the Exchange-Correlation functional."""
        return self._xc_functional

    @xc_functional.setter
    def xc_functional(self, xc_functional: str) -> None:
        """Sets the Exchange-Correlation functional."""
        self._xc_functional = xc_functional

    @property
    def xcf_library(self) -> str:
        """Returns the Exchange-Correlation functional library."""
        return self._xcf_library

    @xcf_library.setter
    def xcf_library(self, xcf_library: str) -> None:
        """Sets the Exchange-Correlation functional library."""
        if xcf_library not in ("libxc", "xcfun"):
            raise QiskitNatureError(
                "Invalid XCF library. It can be either 'libxc' or 'xcfun', not " f"'{xcf_library}'"
            )
        self._xcf_library = xcf_library

    @property
    def conv_tol(self) -> float:
        """Returns the SCF convergence tolerance."""
        return self._conv_tol

    @conv_tol.setter
    def conv_tol(self, conv_tol: float) -> None:
        """Sets the SCF convergence tolerance."""
        self._conv_tol = conv_tol

    @property
    def max_cycle(self) -> int:
        """Returns the maximum number of SCF iterations."""
        return self._max_cycle

    @max_cycle.setter
    def max_cycle(self, max_cycle: int) -> None:
        """Sets the maximum number of SCF iterations."""
        self._max_cycle = max_cycle

    @property
    def init_guess(self) -> str:
        """Returns the method for the initial guess."""
        return self._init_guess

    @init_guess.setter
    def init_guess(self, init_guess: str) -> None:
        """Sets the method for the initial guess."""
        self._init_guess = init_guess

    @property
    def max_memory(self) -> int:
        """Returns the maximum memory allowance for the calculation."""
        return self._max_memory

    @max_memory.setter
    def max_memory(self, max_memory: int) -> None:
        """Sets the maximum memory allowance for the calculation."""
        self._max_memory = max_memory

    @property
    def chkfile(self) -> str:
        """Returns the path to the PySCF checkpoint file."""
        return self._chkfile

    @chkfile.setter
    def chkfile(self, chkfile: str) -> None:
        """Sets the path to the PySCF checkpoint file."""
        self._chkfile = chkfile

    @staticmethod
    def from_molecule(
        molecule: MoleculeInfo,
        *,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        driver_kwargs: dict[str, Any] | None = None,
    ) -> "PySCFDriver":
        """Creates a driver from a molecule.

        Args:
            molecule: the molecular information.
            basis: the basis set.
            method: the SCF method type.
            driver_kwargs: keyword arguments to be passed to driver.

        Returns:
            The constructed driver instance.
        """
        PySCFDriver.check_method_supported(method)
        kwargs = {}
        if driver_kwargs:
            args = inspect.signature(PySCFDriver.__init__).parameters.keys()
            for key, value in driver_kwargs.items():
                if key not in ["self"] and key in args:
                    kwargs[key] = value

        kwargs["atom"] = [
            " ".join(map(str, (name, *coord)))
            for name, coord in zip(molecule.symbols, molecule.coords)
        ]
        kwargs["charge"] = molecule.charge
        kwargs["spin"] = molecule.multiplicity - 1
        kwargs["unit"] = molecule.units
        kwargs["basis"] = PySCFDriver.to_driver_basis(basis)
        kwargs["method"] = method
        return PySCFDriver(**kwargs)

    @staticmethod
    def to_driver_basis(basis: str) -> str:
        """Converts basis to a driver acceptable basis.

        Args:
            basis: The basis set to be used.

        Returns:
            A driver acceptable basis.
        """
        return basis

    @staticmethod
    def check_method_supported(method: MethodType) -> None:
        """Checks that PySCF supports this method.

        Args:
            method: the SCF method type.

        Raises:
            UnsupportMethodError: If the method is not supported.
        """
        # supports all methods
        pass

    def run(self) -> ElectronicStructureProblem:
        """Runs the driver to produce a result.

        Returns:
            ElectronicStructureProblem produced by the run driver.

        Raises:
            QiskitNatureError: if an error during the PySCF setup or calculation occurred.
        """
        self.run_pyscf()
        return self.to_problem()

    def _build_molecule(self) -> None:
        """Builds the PySCF molecule object.

        Raises:
             QiskitNatureError: If building the PySCF molecule object failed.
        """
        # Get config from input parameters
        # molecule is in PySCF atom string format e.g. "H .0 .0 .0; H .0 .0 0.2"
        #          or in Z-Matrix format e.g. "H; O 1 1.08; H 2 1.08 1 107.5"
        # other parameters are as per PySCF got.Mole format
        # pylint: disable=import-error
        from pyscf import gto
        from pyscf.lib import logger as pylogger
        from pyscf.lib import param

        atom = self._check_molecule_format(self.atom)
        if self._max_memory is None:
            self._max_memory = param.MAX_MEMORY

        try:
            verbose = pylogger.QUIET
            output = None
            if logger.isEnabledFor(logging.DEBUG):
                verbose = pylogger.INFO
                file, output = tempfile.mkstemp(suffix=".log")
                os.close(file)

            self._mol = gto.Mole(
                atom=atom,
                unit=self._unit.value,
                basis=self._basis,
                max_memory=self._max_memory,
                verbose=verbose,
                output=output,
            )
            self._mol.symmetry = False
            self._mol.charge = self._charge
            self._mol.spin = self._spin
            self._mol.build(parse_arg=False)

            if output is not None:
                self._process_pyscf_log(output)
                try:
                    os.remove(output)
                except Exception:  # pylint: disable=broad-except
                    pass

        except Exception as exc:
            raise QiskitNatureError("Failed to build the PySCF Molecule object.") from exc

    @staticmethod
    def _check_molecule_format(val: str) -> str | list[str]:
        """Ensures the molecule coordinates are in XYZ format.

        This utility automatically converts a Z-matrix coordinate format into XYZ coordinates.

        Args:
            val: the atomic coordinates.

        Raises:
            QiskitNatureError: If the provided coordinate are badly formatted.

        Returns:
            The coordinates in XYZ format.
        """
        # pylint: disable=import-error
        from pyscf import gto

        atoms = [x.strip() for x in val.split(";")]
        if atoms is None or len(atoms) < 1:
            raise QiskitNatureError("Molecule format error: " + val)

        # An xyz format has 4 parts in each atom, if not then do zmatrix convert
        # Allows dummy atoms, using symbol 'X' in zmatrix format for coord computation to xyz
        parts = [x.strip() for x in atoms[0].split()]
        if len(parts) != 4:
            try:
                newval = []
                for entry in gto.mole.from_zmatrix(val):
                    if entry[0].upper() != "X":
                        newval.append(entry)
                return newval
            except Exception as exc:
                raise QiskitNatureError("Failed to convert atom string: " + val) from exc

        return val

    def run_pyscf(self) -> None:
        """Runs the PySCF calculation.

        This method is part of the public interface to allow the user to easily overwrite it in a
        subclass to further tailor the behavior to some specific use case.

        Raises:
            QiskitNatureError: If an invalid HF method type was supplied.
        """
        self._build_molecule()

        # pylint: disable=import-error
        from pyscf import dft, scf
        from pyscf.lib import chkfile as lib_chkfile

        method_name = None
        method_cls = None
        try:
            # attempt to gather the SCF-method class specified by the MethodType
            method_name = self.method.value.upper()
            method_cls = getattr(scf, method_name)
        except AttributeError as exc:
            raise QiskitNatureError(f"Failed to load {method_name} HF object.") from exc

        self._calc = method_cls(self._mol)

        if method_name in ("RKS", "ROKS", "UKS"):
            self._calc._numint.libxc = getattr(dft, self.xcf_library)
            self._calc.xc = self.xc_functional

        if self._chkfile is not None and os.path.exists(self._chkfile):
            self._calc.__dict__.update(lib_chkfile.load(self._chkfile, "scf"))

            logger.info("PySCF loaded from chkfile e(hf): %s", self._calc.e_tot)
        else:
            self._calc.conv_tol = self._conv_tol
            self._calc.max_cycle = self._max_cycle
            self._calc.init_guess = self._init_guess
            self._calc.kernel()

            logger.info(
                "PySCF kernel() converged: %s, e(hf): %s",
                self._calc.converged,
                self._calc.e_tot,
            )

    def to_qcschema(self, *, include_dipole: bool = True) -> QCSchema:
        # pylint: disable=import-error
        from pyscf import __version__ as pyscf_version
        from pyscf import ao2mo, gto
        from pyscf.tools import dump_mat

        einsum_func, _ = get_einsum()
        data = _QCSchemaData()

        data.mo_coeff, data.mo_coeff_b = self._expand_mo_object(
            self._calc.mo_coeff, array_dimension=3
        )
        data.mo_energy, data.mo_energy_b = self._expand_mo_object(self._calc.mo_energy)
        data.mo_occ, data.mo_occ_b = self._expand_mo_object(self._calc.mo_occ)

        if logger.isEnabledFor(logging.DEBUG):
            # Add some more to PySCF output...
            # First analyze() which prints extra information about MO energy and occupation
            self._mol.stdout.write("\n")
            self._calc.analyze()
            # Now labelled orbitals for contributions to the MOs for s,p,d etc of each atom
            self._mol.stdout.write("\n\n--- Alpha Molecular Orbitals ---\n\n")
            dump_mat.dump_mo(self._mol, data.mo_coeff, digits=7, start=1)
            if data.mo_coeff_b is not None:
                self._mol.stdout.write("\n--- Beta Molecular Orbitals ---\n\n")
                dump_mat.dump_mo(self._mol, data.mo_coeff_b, digits=7, start=1)
            self._mol.stdout.flush()

        data.hij = self._calc.get_hcore()
        data.hij_mo = np.dot(np.dot(data.mo_coeff.T, data.hij), data.mo_coeff)
        if data.mo_coeff_b is not None:
            data.hij_mo_b = np.dot(np.dot(data.mo_coeff_b.T, data.hij), data.mo_coeff_b)

        einsum_ao_to_mo = "pqrs,pi,qj,rk,sl->ijkl"
        if settings.use_symmetry_reduced_integrals:
            data.eri = self._mol.intor("int2e", aosym=8)
            data.eri_mo = fold(ao2mo.full(self._mol, data.mo_coeff, aosym=4))
            if data.mo_coeff_b is not None:
                data.eri_mo_bb = fold(ao2mo.full(self._mol, data.mo_coeff_b, aosym=4))
                data.eri_mo_ba = fold(
                    ao2mo.general(
                        self._mol,
                        [data.mo_coeff_b, data.mo_coeff_b, data.mo_coeff, data.mo_coeff],
                        aosym=4,
                    )
                )
        else:
            data.eri = self._mol.intor("int2e", aosym=1)
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

        data.e_nuc = gto.mole.energy_nuc(self._mol)
        data.e_ref = self._calc.e_tot
        data.symbols = [self._mol.atom_pure_symbol(i) for i in range(self._mol.natm)]
        data.coords = self._mol.atom_coords(unit="Bohr").ravel().tolist()
        data.multiplicity = self._spin + 1
        data.charge = self._charge
        data.masses = list(self._mol.atom_mass_list())
        data.method = self._method.value.upper()
        data.basis = self._basis
        data.creator = "PySCF"
        data.version = pyscf_version
        data.nbasis = self._mol.nbas
        data.nmo = self._mol.nao
        data.nalpha = self._mol.nelec[0]
        data.nbeta = self._mol.nelec[1]

        if include_dipole:
            self._mol.set_common_orig((0, 0, 0))
            ao_dip = self._mol.intor_symmetric("int1e_r", comp=3)

            d_m = self._calc.make_rdm1(self._calc.mo_coeff, self._calc.mo_occ)

            if not (isinstance(d_m, np.ndarray) and d_m.ndim == 2):
                d_m = d_m[0] + d_m[1]

            elec_dip = np.negative(einsum_func("xij,ji->x", ao_dip, d_m).real)
            elec_dip = np.round(elec_dip, decimals=8)
            nucl_dip = einsum_func("i,ix->x", self._mol.atom_charges(), self._mol.atom_coords())
            nucl_dip = np.round(nucl_dip, decimals=8)
            ref_dip = nucl_dip + elec_dip

            logger.info("HF Electronic dipole moment: %s", elec_dip)
            logger.info("Nuclear dipole moment: %s", nucl_dip)
            logger.info("Total dipole moment: %s", ref_dip)
            data.dip_nuc = nucl_dip
            data.dip_ref = ref_dip

            data.dip_x = ao_dip[0]
            data.dip_y = ao_dip[1]
            data.dip_z = ao_dip[2]
            data.dip_mo_x_a = np.dot(np.dot(data.mo_coeff.T, data.dip_x), data.mo_coeff)
            data.dip_mo_y_a = np.dot(np.dot(data.mo_coeff.T, data.dip_y), data.mo_coeff)
            data.dip_mo_z_a = np.dot(np.dot(data.mo_coeff.T, data.dip_z), data.mo_coeff)
            if data.mo_coeff_b is not None:
                data.dip_mo_x_b = np.dot(np.dot(data.mo_coeff_b.T, data.dip_x), data.mo_coeff_b)
                data.dip_mo_y_b = np.dot(np.dot(data.mo_coeff_b.T, data.dip_y), data.mo_coeff_b)
                data.dip_mo_z_b = np.dot(np.dot(data.mo_coeff_b.T, data.dip_z), data.mo_coeff_b)

        return self._to_qcschema(data, include_dipole=include_dipole)

    def to_problem(
        self,
        *,
        basis: ElectronicBasis = ElectronicBasis.MO,
        include_dipole: bool = True,
    ) -> ElectronicStructureProblem:
        qcschema = self.to_qcschema(include_dipole=include_dipole)

        problem = qcschema_to_problem(qcschema, basis=basis, include_dipole=include_dipole)

        if include_dipole and problem.properties.electronic_dipole_moment is not None:
            problem.properties.electronic_dipole_moment.reverse_dipole_sign = True

        return problem

    def _expand_mo_object(
        self,
        mo_object: tuple[np.ndarray | None, np.ndarray | None] | np.ndarray,
        array_dimension: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expands the molecular orbital object into alpha- and beta-spin components.

        Since PySCF 1.6.2, the alpha and beta components are no longer stored as a tuple but as a
        multi-dimensional numpy array. This utility takes care of differentiating these cases.

        Args:
            mo_object: the molecular orbital object to expand.
            array_dimension:  This argument specifies the dimension of the numpy array (if a tuple
                is not encountered). Making this configurable permits this function to be used to
                expand both, MO coefficients (3D array) and MO energies (2D array).

        Returns:
            The (alpha, beta) tuple of MO data.
        """
        if isinstance(mo_object, tuple):
            return mo_object

        if len(mo_object.shape) == array_dimension:
            return mo_object[0], mo_object[1]

        return mo_object, None

    def _process_pyscf_log(self, logfile: str) -> None:
        """Processes a PySCF logfile.

        Args:
            logfile: the path of the PySCF logfile.
        """
        with open(logfile, "r", encoding="utf8") as file:
            contents = file.readlines()

        for i, content in enumerate(contents):
            if content.startswith("System:"):
                contents = contents[i:]
                break

        logger.debug("PySCF processing messages log:\n%s", "".join(contents))
