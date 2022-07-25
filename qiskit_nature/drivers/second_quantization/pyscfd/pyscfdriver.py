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

"""The PySCF Driver."""

import inspect
import logging
import os
import tempfile
import warnings
from enum import Enum
from typing import List, Optional, Tuple, Union, Any, Dict

import numpy as np
from qiskit.utils.validation import validate_min

from qiskit_nature.properties.second_quantization.driver_metadata import DriverMetadata
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicStructureDriverResult,
    AngularMomentum,
    Magnetization,
    ParticleNumber,
    ElectronicEnergy,
    DipoleMoment,
    ElectronicDipoleMoment,
)
from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
import qiskit_nature.optionals as _optionals

from ....exceptions import QiskitNatureError
from ..electronic_structure_driver import ElectronicStructureDriver, MethodType
from ...molecule import Molecule
from ...units_type import UnitsType

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
        atom: Union[str, List[str]] = "H 0.0 0.0 0.0; H 0.0 0.0 0.735",
        unit: UnitsType = UnitsType.ANGSTROM,
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        xc_functional: str = "lda,vwn",
        xcf_library: str = "libxc",
        conv_tol: float = 1e-9,
        max_cycle: int = 50,
        init_guess: InitialGuess = InitialGuess.MINAO,
        max_memory: Optional[int] = None,
        chkfile: Optional[str] = None,
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
                f"`atom` must be either a `str` or `List[str]`, but you passed {atom}"
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
    def atom(self, atom: Union[str, List[str]]) -> None:
        """Sets the atom."""
        if isinstance(atom, list):
            atom = ";".join(atom)
        self._atom = atom.replace("\n", ";")

    @property
    def unit(self) -> UnitsType:
        """Returns the unit."""
        return self._unit

    @unit.setter
    def unit(self, unit: UnitsType) -> None:
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
    @_optionals.HAS_PYSCF.require_in_call
    def from_molecule(
        molecule: Molecule,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        driver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "PySCFDriver":
        """
        Args:
            molecule: molecule
            basis: basis set
            method: Hartree-Fock Method type
            driver_kwargs: kwargs to be passed to driver
        Returns:
            driver
        """
        PySCFDriver.check_method_supported(method)
        kwargs = {}
        if driver_kwargs:
            args = inspect.signature(PySCFDriver.__init__).parameters.keys()
            for key, value in driver_kwargs.items():
                if key not in ["self"] and key in args:
                    kwargs[key] = value

        kwargs["atom"] = [" ".join(map(str, (name, *coord))) for (name, coord) in molecule.geometry]
        kwargs["charge"] = molecule.charge
        kwargs["spin"] = molecule.multiplicity - 1
        kwargs["unit"] = molecule.units
        kwargs["basis"] = PySCFDriver.to_driver_basis(basis)
        kwargs["method"] = method
        return PySCFDriver(**kwargs)

    @staticmethod
    def to_driver_basis(basis: str) -> str:
        """
        Converts basis to a driver acceptable basis
        Args:
            basis: The basis set to be used
        Returns:
            driver acceptable basis
        """
        return basis

    @staticmethod
    def check_method_supported(method: MethodType) -> None:
        """
        Checks that PySCF supports this method.
        Args:
            method: Method type

        Raises:
            UnsupportMethodError: If method not supported.
        """
        # supports all methods
        pass

    def run(self) -> ElectronicStructureDriverResult:
        """
        Returns:
            ElectronicStructureDriverResult produced by the run driver.

        Raises:
            QiskitNatureError: if an error during the PySCF setup or calculation occurred.
        """
        self._build_molecule()
        self.run_pyscf()

        driver_result = self._construct_driver_result()
        return driver_result

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
    def _check_molecule_format(val: str) -> Union[str, List[str]]:
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

    def _construct_driver_result(self) -> ElectronicStructureDriverResult:
        driver_result = ElectronicStructureDriverResult()

        self._populate_driver_result_molecule(driver_result)
        self._populate_driver_result_metadata(driver_result)
        self._populate_driver_result_basis_transform(driver_result)
        self._populate_driver_result_particle_number(driver_result)
        self._populate_driver_result_electronic_energy(driver_result)
        self._populate_driver_result_electronic_dipole_moment(driver_result)

        # TODO: once https://github.com/Qiskit/qiskit-nature/issues/312 is fixed we can stop adding
        # these properties by default.
        # if not settings.dict_aux_operators:
        driver_result.add_property(AngularMomentum(self._mol.nao * 2))
        driver_result.add_property(Magnetization(self._mol.nao * 2))

        return driver_result

    def _populate_driver_result_molecule(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        coords = self._mol.atom_coords(unit="Angstrom")
        geometry = [(self._mol.atom_pure_symbol(i), list(xyz)) for i, xyz in enumerate(coords)]

        driver_result.molecule = Molecule(
            geometry,
            multiplicity=self._spin + 1,
            charge=self._charge,
            masses=list(self._mol.atom_mass_list()),
        )

    def _populate_driver_result_metadata(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        # pylint: disable=import-error
        from pyscf import __version__ as pyscf_version

        cfg = [
            f"atom={self._atom}",
            f"unit={self._unit.value}",
            f"charge={self._charge}",
            f"spin={self._spin}",
            f"basis={self._basis}",
            f"method={self.method.value}",
            f"conv_tol={self._conv_tol}",
            f"max_cycle={self._max_cycle}",
            f"init_guess={self._init_guess}",
            f"max_memory={self._max_memory}",
        ]

        if self.method.value.lower() in ("rks", "roks", "uks"):
            cfg.extend(
                [
                    f"xc_functional={self._xc_functional}",
                    f"xcf_library={self._xcf_library}",
                ]
            )

        driver_result.add_property(DriverMetadata("PYSCF", pyscf_version, "\n".join(cfg + [""])))

    def _populate_driver_result_basis_transform(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        # pylint: disable=import-error
        from pyscf.tools import dump_mat

        mo_coeff, mo_coeff_b = self._extract_mo_data("mo_coeff", array_dimension=3)

        if logger.isEnabledFor(logging.DEBUG):
            # Add some more to PySCF output...
            # First analyze() which prints extra information about MO energy and occupation
            self._mol.stdout.write("\n")
            self._calc.analyze()
            # Now labelled orbitals for contributions to the MOs for s,p,d etc of each atom
            self._mol.stdout.write("\n\n--- Alpha Molecular Orbitals ---\n\n")
            dump_mat.dump_mo(self._mol, mo_coeff, digits=7, start=1)
            if mo_coeff_b is not None:
                self._mol.stdout.write("\n--- Beta Molecular Orbitals ---\n\n")
                dump_mat.dump_mo(self._mol, mo_coeff_b, digits=7, start=1)
            self._mol.stdout.flush()

        driver_result.add_property(
            ElectronicBasisTransform(
                ElectronicBasis.AO,
                ElectronicBasis.MO,
                mo_coeff,
                mo_coeff_b,
            )
        )

    def _populate_driver_result_particle_number(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        mo_occ, mo_occ_b = self._extract_mo_data("mo_occ")

        driver_result.add_property(
            ParticleNumber(
                num_spin_orbitals=self._mol.nao * 2,
                num_particles=(self._mol.nelec[0], self._mol.nelec[1]),
                occupation=mo_occ,
                occupation_beta=mo_occ_b,
            )
        )

    def _populate_driver_result_electronic_energy(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        # pylint: disable=import-error
        from pyscf import gto

        basis_transform = driver_result.get_property(ElectronicBasisTransform)

        one_body_ao = OneBodyElectronicIntegrals(
            ElectronicBasis.AO,
            (self._calc.get_hcore(), None),
        )

        two_body_ao = TwoBodyElectronicIntegrals(
            ElectronicBasis.AO,
            (self._mol.intor("int2e", aosym=1), None, None, None),
        )

        one_body_mo = one_body_ao.transform_basis(basis_transform)
        two_body_mo = two_body_ao.transform_basis(basis_transform)

        electronic_energy = ElectronicEnergy(
            [one_body_ao, two_body_ao, one_body_mo, two_body_mo],
            nuclear_repulsion_energy=gto.mole.energy_nuc(self._mol),
            reference_energy=self._calc.e_tot,
        )

        electronic_energy.kinetic = OneBodyElectronicIntegrals(
            ElectronicBasis.AO,
            (self._mol.intor_symmetric("int1e_kin"), None),
        )
        electronic_energy.overlap = OneBodyElectronicIntegrals(
            ElectronicBasis.AO,
            (self._calc.get_ovlp(), None),
        )

        orbs_energy, orbs_energy_b = self._extract_mo_data("mo_energy")
        orbital_energies = (
            (orbs_energy, orbs_energy_b) if orbs_energy_b is not None else orbs_energy
        )
        electronic_energy.orbital_energies = np.asarray(orbital_energies)

        driver_result.add_property(electronic_energy)

    def _populate_driver_result_electronic_dipole_moment(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        basis_transform = driver_result.get_property(ElectronicBasisTransform)

        self._mol.set_common_orig((0, 0, 0))
        ao_dip = self._mol.intor_symmetric("int1e_r", comp=3)

        d_m = self._calc.make_rdm1(self._calc.mo_coeff, self._calc.mo_occ)

        if not (isinstance(d_m, np.ndarray) and d_m.ndim == 2):
            d_m = d_m[0] + d_m[1]

        elec_dip = np.negative(np.einsum("xij,ji->x", ao_dip, d_m).real)
        elec_dip = np.round(elec_dip, decimals=8)
        nucl_dip = np.einsum("i,ix->x", self._mol.atom_charges(), self._mol.atom_coords())
        nucl_dip = np.round(nucl_dip, decimals=8)

        logger.info("HF Electronic dipole moment: %s", elec_dip)
        logger.info("Nuclear dipole moment: %s", nucl_dip)
        logger.info("Total dipole moment: %s", nucl_dip + elec_dip)

        x_dip_ints = OneBodyElectronicIntegrals(ElectronicBasis.AO, (ao_dip[0], None))
        y_dip_ints = OneBodyElectronicIntegrals(ElectronicBasis.AO, (ao_dip[1], None))
        z_dip_ints = OneBodyElectronicIntegrals(ElectronicBasis.AO, (ao_dip[2], None))

        x_dipole = DipoleMoment("x", [x_dip_ints, x_dip_ints.transform_basis(basis_transform)])
        y_dipole = DipoleMoment("y", [y_dip_ints, y_dip_ints.transform_basis(basis_transform)])
        z_dipole = DipoleMoment("z", [z_dip_ints, z_dip_ints.transform_basis(basis_transform)])

        driver_result.add_property(
            ElectronicDipoleMoment(
                [x_dipole, y_dipole, z_dipole],
                nuclear_dipole_moment=nucl_dip,
                reverse_dipole_sign=True,
            )
        )

    def _extract_mo_data(
        self, name: str, array_dimension: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract molecular orbital data from a PySCF calculation object.

        Args:
            name: the name of the molecular orbital data field to extract.
            array_dimension: since PySCF 1.6.2, the alpha and beta components are no longer stored
                as a tuple but as a multi-dimensional numpy array. This argument specifies the
                dimension of that array in such a case. Making this configurable permits this
                function to be used to extract both, MO coefficients (3D array) and MO energies (2D
                array).

        Returns:
            The (alpha, beta) tuple of MO data.
        """
        attr = getattr(self._calc, name)
        if isinstance(attr, tuple):
            attr_alpha = attr[0]
            attr_beta = attr[1]
        else:
            # Since PySCF 1.6.2, instead of a tuple it could be a multi-dimensional array with the
            # first dimension indexing the arrays for alpha and beta
            if len(attr.shape) == array_dimension:
                attr_alpha = attr[0]
                attr_beta = attr[1]
            else:
                attr_alpha = attr
                attr_beta = None
        return attr_alpha, attr_beta

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
