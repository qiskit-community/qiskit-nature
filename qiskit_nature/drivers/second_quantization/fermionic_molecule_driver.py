# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module implements a common molecule fermionic based driver.
"""

from typing import Optional, Dict, Any, Type
import logging
from enum import Enum

from qiskit.exceptions import MissingOptionalLibraryError
from .fermionic_driver import FermionicDriver, MethodType
from ..molecule import Molecule
from .pyscfd import PySCFDriver
from .psi4d import PSI4Driver
from .pyquanted import PyQuanteDriver
from .gaussiand import GaussianDriver

logger = logging.getLogger(__name__)


class FermionicDriverType(Enum):
    """Fermionic Driver Type."""

    AUTO = "auto"
    PYSCF = "pyscf"
    PSI4 = "psi4"
    PYQUANTE = "pyquante"
    GAUSSIAN = "gaussian"

    @staticmethod
    def driver_class_from_type(driver_type: "FermionicDriverType") -> Type[FermionicDriver]:
        """
        Get driver class from driver type

        Args:
            driver_type:type of driver to be used. If `AUTO` is selected, it will use
                        the first driver installed in the following order:
                        `PYSCF`, `PSI4`, `PYQUANTE`, `GAUSSIAN`

        Returns:
            driver class

        Raises:
            MissingOptionalLibraryError: Driver not installed.
        """
        driver_class = None
        if driver_type == FermionicDriverType.AUTO:
            missing_error = None
            for item in FermionicDriverType:
                if item != FermionicDriverType.AUTO:
                    try:
                        driver_class = FermionicDriverType.driver_class_from_type(item)
                        break
                    except MissingOptionalLibraryError as ex:
                        if missing_error is None:
                            missing_error = ex
            if driver_class is None:
                raise missing_error
        elif driver_type == FermionicDriverType.PYSCF:
            PySCFDriver.check_installed()
            driver_class = PySCFDriver
        elif driver_type == FermionicDriverType.PSI4:
            PSI4Driver.check_installed()
            driver_class = PSI4Driver
        elif driver_type == FermionicDriverType.PYQUANTE:
            PyQuanteDriver.check_installed()
            driver_class = PyQuanteDriver
        elif driver_type == FermionicDriverType.GAUSSIAN:
            GaussianDriver.check_installed()
            driver_class = GaussianDriver
        else:
            MissingOptionalLibraryError(libname=driver_type, name="FermionicDriverType")

        logger.debug("%s found from type %s.", driver_class.__name__, driver_type.value)
        return driver_class


class FermionicMoleculeDriver(FermionicDriver):
    """
    Molecule based fermionic driver
    """

    def __init__(
        self,
        molecule: Molecule,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        driver_type: FermionicDriverType = FermionicDriverType.AUTO,
        driver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            molecule: molecule
            basis: basis set
            method: The method type to be used for the calculation.
            driver_type:type of driver to be used. If `AUTO` is selected, it will use
                        the first driver installed in the following order:
                        `PYSCF`, `PSI4`, `PYQUANTE`, `GAUSSIAN`
            driver_kwargs: kwargs to be passed to driver

        Raises:
            MissingOptionalLibraryError: Driver not installed.
        """
        super().__init__()
        self._driver_class = FermionicDriverType.driver_class_from_type(driver_type)
        self._driver_kwargs = driver_kwargs
        self._molecule = molecule
        self._basis = basis
        self._method = method

    @property
    def molecule(self) -> Optional[Molecule]:
        """return molecule"""
        return self._molecule

    @molecule.setter
    def molecule(self, value: Molecule) -> None:
        """set molecule"""
        self._molecule = value

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
        """return Hartree-Fock method"""
        return self._method

    @method.setter
    def method(self, value: MethodType) -> None:
        """set Hartree-Fock method"""
        self._method = value

    @property
    def driver_kwargs(self) -> Optional[Dict[str, Any]]:
        """return driver kwargs"""
        return self._driver_kwargs

    @driver_kwargs.setter
    def driver_kwargs(self, value: Optional[Dict[str, Any]]) -> None:
        """set driver kwargs"""
        self._driver_kwargs = value

    def run(self):
        """
        Runs a driver to produce an output data structure.
        """
        driver = self._driver_class.from_molecule(
            self.molecule, self.basis, self.method, self.driver_kwargs
        )
        return driver.run()
