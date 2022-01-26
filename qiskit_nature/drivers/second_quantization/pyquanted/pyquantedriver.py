# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" PyQuante Driver """

import importlib
import inspect
import logging
from enum import Enum
from typing import Union, List, Optional, Any, Dict

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.utils.validation import validate_min

from qiskit_nature import QiskitNatureError
from qiskit_nature.properties.second_quantization.electronic import ElectronicStructureDriverResult

from .integrals import compute_integrals
from ..electronic_structure_driver import ElectronicStructureDriver, MethodType
from ...molecule import Molecule
from ...units_type import UnitsType
from ....exceptions import UnsupportMethodError

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
        validate_min("maxiters", maxiters, 1)
        PyQuanteDriver.check_installed()
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

    @staticmethod
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
        PyQuanteDriver.check_installed()
        PyQuanteDriver.check_method_supported(method)
        kwargs = {}
        if driver_kwargs:
            args = inspect.getfullargspec(PyQuanteDriver.__init__).args
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
    def check_installed() -> None:
        """
        Checks if PyQuante is installed and available

        Raises:
            MissingOptionalLibraryError: if not installed.
        """
        try:
            spec = importlib.util.find_spec("pyquante2")
            if spec is not None:
                return
        except Exception as ex:
            logger.debug("PyQuante2 check error %s", str(ex))
            raise MissingOptionalLibraryError(
                libname="PyQuante2",
                name="PyQuanteDriver",
                msg="See https://github.com/rpmuller/pyquante2",
            ) from ex

        raise MissingOptionalLibraryError(
            libname="PyQuante2",
            name="PyQuanteDriver",
            msg="See https://github.com/rpmuller/pyquante2",
        )

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
        atoms = self._atoms
        charge = self._charge
        multiplicity = self._multiplicity
        units = self._units
        basis = self.basis
        method = self.method

        q_mol = compute_integrals(
            atoms=atoms,
            units=units.value,
            charge=charge,
            multiplicity=multiplicity,
            basis=basis.value,
            method=method.value,
            tol=self._tol,
            maxiters=self._maxiters,
        )

        q_mol.origin_driver_name = "PYQUANTE"
        cfg = [
            f"atoms={atoms}",
            f"units={units.value}",
            f"charge={charge}",
            f"multiplicity={multiplicity}",
            f"basis={basis.value}",
            f"method={method.value}",
            f"tol={self._tol}",
            f"maxiters={self._maxiters}",
            "",
        ]
        q_mol.origin_driver_config = "\n".join(cfg)

        return ElectronicStructureDriverResult.from_legacy_driver_result(q_mol)
