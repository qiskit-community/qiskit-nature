# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module implements a common molecule electronic structure based driver.
"""

from typing import Optional, Dict, Any, Type
import logging
import importlib
from enum import Enum

from qiskit.exceptions import MissingOptionalLibraryError

from qiskit_nature.exceptions import UnsupportMethodError
from qiskit_nature.second_q.operator_factories.electronic import (
    ElectronicStructureDriverResult,
)
from .electronic_structure_driver import ElectronicStructureDriver, MethodType
from ..molecule import Molecule

logger = logging.getLogger(__name__)


class ElectronicStructureDriverType(Enum):
    """Electronic structure Driver Type."""

    AUTO = "auto"
    PYSCF = "PySCFDriver"
    PSI4 = "PSI4Driver"
    PYQUANTE = "PyQuanteDriver"
    GAUSSIAN = "GaussianDriver"

    @staticmethod
    def driver_class_from_type(
        driver_type: "ElectronicStructureDriverType",
        method: MethodType,
    ) -> Type[ElectronicStructureDriver]:
        """
        Get driver class from driver type

        Args:
            driver_type: type of driver to be used. If `AUTO` is selected, it will use
                        the first driver installed and that supports the given method
                        in the following order
                        `PYSCF`, `PSI4`, `PYQUANTE`, `GAUSSIAN`
            method: Used to verify if the driver supports it.

        Returns:
            driver class

        Raises:
            MissingOptionalLibraryError: Driver not installed.
            UnsupportMethodError: method not supported by driver.
        """
        driver_class = None
        if driver_type == ElectronicStructureDriverType.AUTO:
            error = None
            for item in ElectronicStructureDriverType:
                if item != ElectronicStructureDriverType.AUTO:
                    try:
                        driver_class = ElectronicStructureDriverType.driver_class_from_type(
                            item, method
                        )
                        break
                    except (MissingOptionalLibraryError, UnsupportMethodError) as ex:
                        if error is None:
                            error = ex
            if driver_class is None:
                raise error
        else:
            driver_module = importlib.import_module("qiskit_nature.drivers.second_q")
            class_obj = getattr(driver_module, driver_type.value, None)
            if class_obj is None:
                raise MissingOptionalLibraryError(
                    libname=driver_type, name="ElectronicStructureDriverType"
                )
            # instantiating the object will check if the driver is installed
            _ = class_obj()
            class_obj.check_method_supported(method)
            driver_class = class_obj

        logger.debug("%s found from type %s.", driver_class.__name__, driver_type.value)
        return driver_class


class ElectronicStructureMoleculeDriver(ElectronicStructureDriver):
    """
    Molecule based electronic structure driver
    """

    def __init__(
        self,
        molecule: Molecule,
        basis: str = "sto3g",
        method: MethodType = MethodType.RHF,
        driver_type: ElectronicStructureDriverType = ElectronicStructureDriverType.AUTO,
        driver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            molecule: molecule
            basis: basis set
            method: The method type to be used for the calculation.
            driver_type:type of driver to be used. If `AUTO` is selected, it will use
                        the first driver installed and that supports the given method
                        in the following order:
                        `PYSCF`, `PSI4`, `PYQUANTE`, `GAUSSIAN`
            driver_kwargs: kwargs to be passed to driver
        """
        self._molecule = molecule
        self._basis = basis
        self._method = method
        self._driver_type = driver_type
        self._driver_kwargs = driver_kwargs

    @property
    def molecule(self) -> Optional[Molecule]:
        """return molecule"""
        return self._molecule

    @molecule.setter
    def molecule(self, value: Molecule) -> None:
        """set molecule"""
        self._molecule = value

    @property
    def driver_type(self) -> ElectronicStructureDriverType:
        """return driver type"""
        return self._driver_type

    @driver_type.setter
    def driver_type(self, value: ElectronicStructureDriverType) -> None:
        """set driver type"""
        self._driver_type = value

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

    def run(self) -> ElectronicStructureDriverResult:
        driver_class = ElectronicStructureDriverType.driver_class_from_type(
            self.driver_type, self.method
        )
        driver = driver_class.from_molecule(  # type: ignore
            self.molecule, self.basis, self.method, self.driver_kwargs
        )
        return driver.run()
