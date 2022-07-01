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
This module implements a common molecule vibrational structure based driver.
"""

from typing import Optional, Dict, Any, Type
import logging
import importlib
from enum import Enum

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_nature.second_q.operator_factories.vibrational import (
    VibrationalStructureDriverResult,
)
from .vibrational_structure_driver import VibrationalStructureDriver
from .molecule import Molecule

logger = logging.getLogger(__name__)


class VibrationalStructureDriverType(Enum):
    """Vibrational structure Driver Type."""

    AUTO = "auto"
    GAUSSIAN_FORCES = "GaussianForcesDriver"

    @staticmethod
    def driver_class_from_type(
        driver_type: "VibrationalStructureDriverType",
    ) -> Type[VibrationalStructureDriver]:
        """
        Get driver class from driver type

        Args:
            driver_type:type of driver to be used. If `AUTO` is selected, it will use
                        the first driver installed in the following order:
                        `GAUSSIAN_FORCES`

        Returns:
            driver class

        Raises:
            MissingOptionalLibraryError: Driver not installed.
        """
        driver_class = None
        if driver_type == VibrationalStructureDriverType.AUTO:
            missing_error = None
            for item in VibrationalStructureDriverType:
                if item != VibrationalStructureDriverType.AUTO:
                    try:
                        driver_class = VibrationalStructureDriverType.driver_class_from_type(item)
                        break
                    except MissingOptionalLibraryError as ex:
                        if missing_error is None:
                            missing_error = ex
            if driver_class is None:
                raise missing_error
        else:
            driver_module = importlib.import_module("qiskit_nature.drivers.second_q")
            driver_class = getattr(driver_module, driver_type.value, None)
            if driver_class is None:
                raise MissingOptionalLibraryError(
                    libname=driver_type, name="VibrationalStructureDriverType"
                )
            # instantiating the object will check if the driver is installed
            _ = driver_class()

        logger.debug("%s found from type %s.", driver_class.__name__, driver_type.value)
        return driver_class


class VibrationalStructureMoleculeDriver(VibrationalStructureDriver):
    """
    Molecule based vibrational structure driver
    """

    def __init__(
        self,
        molecule: Molecule,
        basis: str = "sto3g",
        driver_type: VibrationalStructureDriverType = VibrationalStructureDriverType.AUTO,
        driver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            molecule: molecule
            basis: basis set
            driver_type:type of driver to be used. If `AUTO` is selected, it will use
                        the first driver installed in the following order:
                        `GAUSSIAN_FORCES`
            driver_kwargs: kwargs to be passed to driver
        """
        self._molecule = molecule
        self._basis = basis
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
    def driver_type(self) -> VibrationalStructureDriverType:
        """return driver type"""
        return self._driver_type

    @driver_type.setter
    def driver_type(self, value: VibrationalStructureDriverType) -> None:
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
    def driver_kwargs(self) -> Optional[Dict[str, Any]]:
        """return driver kwargs"""
        return self._driver_kwargs

    @driver_kwargs.setter
    def driver_kwargs(self, value: Optional[Dict[str, Any]]) -> None:
        """set driver kwargs"""
        self._driver_kwargs = value

    def run(self) -> VibrationalStructureDriverResult:
        driver_class = VibrationalStructureDriverType.driver_class_from_type(self.driver_type)
        driver = driver_class.from_molecule(  # type: ignore
            self.molecule, self.basis, self.driver_kwargs
        )
        return driver.run()
