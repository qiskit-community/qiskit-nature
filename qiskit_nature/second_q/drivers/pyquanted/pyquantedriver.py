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

""" PyQuante Driver """

import inspect
import logging
import re
from enum import Enum
from typing import Union, List, Optional, Any, Dict, Tuple

import numpy as np

from qiskit.utils.validation import validate_min

from qiskit_nature import QiskitNatureError
from qiskit_nature.constants import BOHR, PERIODIC_TABLE
from qiskit_nature.exceptions import UnsupportMethodError
from qiskit_nature.second_q.properties.driver_metadata import DriverMetadata
from qiskit_nature.second_q.properties import (
    ElectronicStructureDriverResult,
    AngularMomentum,
    Magnetization,
    ParticleNumber,
    ElectronicEnergy,
)
from qiskit_nature.second_q.properties.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.second_q.properties.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
import qiskit_nature.optionals as _optionals

from ..electronic_structure_driver import ElectronicStructureDriver, MethodType
from ..molecule import Molecule
from ..units_type import UnitsType

logger = logging.getLogger(__name__)


class BasisType(Enum):
    """Basis Type"""

    BSTO3G = "sto3g"
    B631G = "6-31g"
    B631GSS = "6-31g**"

    @staticmethod
    def type_from_string(basis: str) -> "BasisType":
        """
        Get basis type from string
        Args:
            basis: The basis set to be used
        Returns:
            BasisType basis
        Raises:
            QiskitNatureError: invalid basis
        """
        for item in BasisType:
            if basis == item.value:
                return item
        raise QiskitNatureError(f"Invalid Basis type basis {basis}.")


@_optionals.HAS_PYQUANTE2.require_in_instance
class PyQuanteDriver(ElectronicStructureDriver):
    """
    Qiskit Nature driver using the PyQuante2 library.

    See https://github.com/rpmuller/pyquante2
    """

    def __init__(
        self,
        atoms: Union[str, List[str]] = "H 0.0 0.0 0.0; H 0.0 0.0 0.735",
        units: UnitsType = UnitsType.ANGSTROM,
        charge: int = 0,
        multiplicity: int = 1,
        basis: BasisType = BasisType.BSTO3G,
        method: MethodType = MethodType.RHF,
        tol: float = 1e-8,
        maxiters: int = 100,
    ) -> None:
        """
        Args:
            atoms: Atoms list or string separated by semicolons or line breaks. Each element in the
                list is an atom followed by position e.g. `H 0.0 0.0 0.5`. The preceding example
                shows the `XYZ` format for position but `Z-Matrix` format is supported too here.
            units: Angstrom or Bohr.
            charge: Charge on the molecule.
            multiplicity: Spin multiplicity (2S+1)
            basis: Basis set; sto3g, 6-31g or 6-31g**
            method: Hartree-Fock Method type.
            tol: Convergence tolerance see pyquante2.scf hamiltonians and iterators
            maxiters: Convergence max iterations see pyquante2.scf hamiltonians and iterators,
                has a min. value of 1.

        Raises:
            QiskitNatureError: Invalid Input
        """
        super().__init__()
        # pylint: disable=import-error
        from pyquante2 import molecule as pyquante_molecule
        from pyquante2 import rhf, uhf, rohf, basisset

        validate_min("maxiters", maxiters, 1)
        PyQuanteDriver.check_method_supported(method)
        if not isinstance(atoms, str) and not isinstance(atoms, list):
            raise QiskitNatureError(f"Invalid atom input for PYQUANTE Driver '{atoms}'")

        if isinstance(atoms, list):
            atoms = ";".join(atoms)
        elif isinstance(atoms, str):
            atoms = atoms.replace("\n", ";")

        self._atoms = atoms
        self._units = units
        self._charge = charge
        self._multiplicity = multiplicity
        self._basis = basis
        self._method = method
        self._tol = tol
        self._maxiters = maxiters

        self._mol: pyquante_molecule = None
        self._bfs: basisset = None
        self._calc: Union[rhf, rohf, uhf] = None
        self._nmo: int = None

    @property
    def atoms(self) -> str:
        """Returns the atom."""
        return self._atoms

    @atoms.setter
    def atoms(self, atom: Union[str, List[str]]) -> None:
        """Sets the atom."""
        if isinstance(atom, list):
            atom = ";".join(atom)
        self._atoms = atom.replace("\n", ";")

    @property
    def units(self) -> UnitsType:
        """Returns the units."""
        return self._units

    @units.setter
    def units(self, units: UnitsType) -> None:
        """Sets the units."""
        self._units = units

    @property
    def charge(self) -> int:
        """Returns the charge."""
        return self._charge

    @charge.setter
    def charge(self, charge: int) -> None:
        """Sets the charge."""
        self._charge = charge

    @property
    def multiplicity(self) -> int:
        """Returns the multiplicity."""
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, multiplicity: int) -> None:
        """Sets the multiplicity."""
        self._multiplicity = multiplicity

    @property
    def basis(self) -> BasisType:
        """return basis"""
        return self._basis

    @basis.setter
    def basis(self, value: BasisType) -> None:
        """set basis"""
        self._basis = value

    @property
    def method(self) -> MethodType:
        """return Hartree-Fock method"""
        return self._method

    @method.setter
    def method(self, value: MethodType) -> None:
        """set Hartree-Fock method"""
        self._method = value

    @property
    def tol(self) -> float:
        """Returns the SCF convergence tolerance."""
        return self._tol

    @tol.setter
    def tol(self, tol: float) -> None:
        """Sets the SCF convergence tolerance."""
        self._tol = tol

    @property
    def maxiters(self) -> int:
        """Returns the maximum number of SCF iterations."""
        return self._maxiters

    @maxiters.setter
    def maxiters(self, maxiters: int) -> None:
        """Sets the maximum number of SCF iterations."""
        self._maxiters = maxiters

    @staticmethod
    @_optionals.HAS_PYQUANTE2.require_in_call
    def from_molecule(
        molecule: Molecule,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        driver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "PyQuanteDriver":
        """
        Args:
            molecule: molecule
            basis: basis set
            method: Hartree-Fock Method type
            driver_kwargs: kwargs to be passed to driver
        Returns:
            driver
        """
        PyQuanteDriver.check_method_supported(method)
        kwargs = {}
        if driver_kwargs:
            args = inspect.signature(PyQuanteDriver.__init__).parameters.keys()
            for key, value in driver_kwargs.items():
                if key not in ["self"] and key in args:
                    kwargs[key] = value

        kwargs["atoms"] = ";".join(
            [name + " " + " ".join(map(str, coord)) for (name, coord) in molecule.geometry]
        )
        kwargs["charge"] = molecule.charge
        kwargs["multiplicity"] = molecule.multiplicity
        kwargs["units"] = molecule.units
        kwargs["basis"] = PyQuanteDriver.to_driver_basis(basis)
        kwargs["method"] = method
        return PyQuanteDriver(**kwargs)

    @staticmethod
    def to_driver_basis(basis: str) -> BasisType:
        """
        Converts basis to a driver acceptable basis
        Args:
            basis: The basis set to be used
        Returns:
            driver acceptable basis
        """
        return BasisType.type_from_string(basis)

    @staticmethod
    def check_method_supported(method: MethodType) -> None:
        """
        Checks that PyQuante supports this method.
        Args:
            method: Method type

        Raises:
            UnsupportMethodError: If method not supported.
        """
        if method not in [MethodType.RHF, MethodType.ROHF, MethodType.UHF]:
            raise UnsupportMethodError(f"Invalid Pyquante method {method.value}.")

    def run(self) -> ElectronicStructureDriverResult:
        """
        Returns:
            ElectronicStructureDriverResult produced by the run driver.

        Raises:
            QiskitNatureError: if an error during the PyQuante setup or calculation occurred.
        """
        self._build_molecule()
        self.run_pyquante()

        driver_result = self._construct_driver_result()
        return driver_result

    def _build_molecule(self) -> None:
        # pylint: disable=import-error
        from pyquante2 import molecule as pyquante_molecule

        atoms = self._check_molecule_format(self.atoms)

        parts = [x.strip() for x in atoms.split(";")]
        if parts is None or len(parts) < 1:
            raise QiskitNatureError("Molecule format error: " + atoms)
        geom = []
        for part in parts:
            geom.append(self._parse_atom(part))

        if len(geom) < 1:
            raise QiskitNatureError("Molecule format error: " + atoms)

        self._mol = pyquante_molecule(
            geom, units=self.units.value, charge=self.charge, multiplicity=self.multiplicity
        )

    @staticmethod
    def _check_molecule_format(val: str) -> str:
        """If it seems to be zmatrix rather than xyz format we convert before returning"""
        # pylint: disable=import-error
        from pyquante2.geo.zmatrix import z2xyz

        atoms = [x.strip() for x in val.split(";")]
        if atoms is None or len(atoms) < 1:
            raise QiskitNatureError("Molecule format error: " + val)

        # An xyz format has 4 parts in each atom, if not then do zmatrix convert
        # Allows dummy atoms, using symbol 'X' in zmatrix format for coord computation to xyz
        parts = [x.strip() for x in atoms[0].split()]
        if len(parts) != 4:
            try:
                zmat = []
                for atom in atoms:
                    parts = [x.strip() for x in atom.split()]
                    z: List[Union[str, int, float]] = [parts[0]]
                    for i in range(1, len(parts), 2):
                        z.append(int(parts[i]))
                        z.append(float(parts[i + 1]))
                    zmat.append(z)
                xyz = z2xyz(zmat)
                new_val = ""
                for atm in xyz:
                    if atm[0].upper() == "X":
                        continue
                    if new_val:
                        new_val += "; "
                    new_val += f"{atm[0]} {atm[1]} {atm[2]} {atm[3]}"
                return new_val
            except Exception as exc:
                raise QiskitNatureError("Failed to convert atom string: " + val) from exc

        return val

    @staticmethod
    def _parse_atom(val: str) -> Tuple[int, float, float, float]:
        if val is None or len(val) < 1:
            raise QiskitNatureError("Molecule atom format error: empty")

        parts = re.split(r"\s+", val)
        if len(parts) != 4:
            raise QiskitNatureError("Molecule atom format error: " + val)

        parts[0] = parts[0].lower().capitalize()
        if not parts[0].isdigit():
            if parts[0] in PERIODIC_TABLE:
                parts[0] = PERIODIC_TABLE.index(parts[0])
            else:
                raise QiskitNatureError("Molecule atom symbol error: " + parts[0])

        return int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3])

    def run_pyquante(self):
        """Runs the PyQuante calculation.

        This method is part of the public interface to allow the user to easily overwrite it in a
        subclass to further tailor the behavior to some specific use case.

        Raises:
            QiskitNatureError: If an invalid HF method type was supplied.
        """
        # pylint: disable=import-error
        from pyquante2 import rhf, uhf, rohf, basisset

        self._bfs = basisset(self._mol, self.basis.value)

        if self.method == MethodType.RHF:
            self._calc = rhf(self._mol, self._bfs)
        elif self.method == MethodType.ROHF:
            self._calc = rohf(self._mol, self._bfs)
        elif self.method == MethodType.UHF:
            self._calc = uhf(self._mol, self._bfs)
        else:
            raise QiskitNatureError(f"Invalid method type: {self.method}")

        self._calc.converge(tol=self.tol, maxiters=self.maxiters)
        logger.debug("PyQuante2 processing information:\n%s", self._calc)

    def _construct_driver_result(self) -> ElectronicStructureDriverResult:
        driver_result = ElectronicStructureDriverResult()

        self._populate_driver_result_molecule(driver_result)
        self._populate_driver_result_metadata(driver_result)
        self._populate_driver_result_basis_transform(driver_result)
        self._populate_driver_result_particle_number(driver_result)
        self._populate_driver_result_electronic_energy(driver_result)

        # TODO: once https://github.com/Qiskit/qiskit-nature/issues/312 is fixed we can stop adding
        # these properties by default.
        # if not settings.dict_aux_operators:
        driver_result.add_property(AngularMomentum(self._nmo * 2))
        driver_result.add_property(Magnetization(self._nmo * 2))

        return driver_result

    def _populate_driver_result_molecule(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        geometry: List[Tuple[str, List[float]]] = []
        for atom in self._mol.atoms:
            atuple = atom.atuple()
            geometry.append((PERIODIC_TABLE[atuple[0]], [a * BOHR for a in atuple[1:]]))

        driver_result.molecule = Molecule(
            geometry, multiplicity=self._mol.multiplicity, charge=self._mol.charge
        )

    def _populate_driver_result_metadata(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        cfg = [
            f"atoms={self.atoms}",
            f"units={self.units.value}",
            f"charge={self.charge}",
            f"multiplicity={self.multiplicity}",
            f"basis={self.basis.value}",
            f"method={self.method.value}",
            f"tol={self._tol}",
            f"maxiters={self._maxiters}",
            "",
        ]

        driver_result.add_property(DriverMetadata("PYQUANTE", "?", "\n".join(cfg)))

    def _populate_driver_result_basis_transform(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        if hasattr(self._calc, "orbs"):
            mo_coeff = self._calc.orbs
            mo_coeff_b = None
        else:
            mo_coeff = self._calc.orbsa
            mo_coeff_b = self._calc.orbsb

        self._nmo = len(mo_coeff)

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
        driver_result.add_property(
            ParticleNumber(
                num_spin_orbitals=self._nmo * 2,
                num_particles=(self._mol.nup(), self._mol.ndown()),
            )
        )

    def _populate_driver_result_electronic_energy(
        self, driver_result: ElectronicStructureDriverResult
    ) -> None:
        # pylint: disable=import-error
        from pyquante2 import onee_integrals
        from pyquante2.ints.integrals import twoe_integrals

        basis_transform = driver_result.get_property(ElectronicBasisTransform)

        integrals = onee_integrals(self._bfs, self._mol)

        hij = integrals.T + integrals.V
        hijkl = twoe_integrals(self._bfs)

        one_body_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (hij, None))

        two_body_ao = TwoBodyElectronicIntegrals(
            ElectronicBasis.AO,
            (hijkl.transform(np.identity(self._nmo)), None, None, None),
        )

        one_body_mo = one_body_ao.transform_basis(basis_transform)
        two_body_mo = two_body_ao.transform_basis(basis_transform)

        electronic_energy = ElectronicEnergy(
            [one_body_ao, two_body_ao, one_body_mo, two_body_mo],
            nuclear_repulsion_energy=self._mol.nuclear_repulsion(),
            reference_energy=self._calc.energy,
        )

        if hasattr(self._calc, "orbe"):
            orbs_energy = self._calc.orbe
            orbs_energy_b = None
        else:
            orbs_energy = self._calc.orbea
            orbs_energy_b = self._calc.orbeb

        orbital_energies = (
            (orbs_energy, orbs_energy_b) if orbs_energy_b is not None else orbs_energy
        )
        electronic_energy.orbital_energies = np.asarray(orbital_energies)

        electronic_energy.kinetic = OneBodyElectronicIntegrals(
            ElectronicBasis.AO, (integrals.T, None)
        )
        electronic_energy.overlap = OneBodyElectronicIntegrals(
            ElectronicBasis.AO, (integrals.S, None)
        )

        driver_result.add_property(electronic_energy)
