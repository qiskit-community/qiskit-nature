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
This module implements a common molecule bosonic based driver.
"""

from typing import Optional, Dict, Any, Type
import logging
from enum import Enum

from qiskit.exceptions import MissingOptionalLibraryError
from .bosonic_driver import BosonicDriver
from ..molecule import Molecule
from .gaussiand import GaussianForcesDriver

logger = logging.getLogger(__name__)


class BosonicDriverType(Enum):
    """Bosonic Driver Type."""

    AUTO = "auto"
    GAUSSIAN_FORCES = "gaussian_forces"

    @staticmethod
    def driver_class_from_type(driver_type: "BosonicDriverType") -> Type[BosonicDriver]:
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
        if driver_type == BosonicDriverType.AUTO:
            missing_error = None
            for item in BosonicDriverType:
                if item != BosonicDriverType.AUTO:
                    try:
                        driver_class = BosonicDriverType.driver_class_from_type(item)
                        break
                    except MissingOptionalLibraryError as ex:
                        if missing_error is None:
                            missing_error = ex
            if driver_class is None:
                raise missing_error
        elif driver_type == BosonicDriverType.GAUSSIAN_FORCES:
            GaussianForcesDriver.check_installed()
            driver_class = GaussianForcesDriver

        logger.debug("%s found from type %s.", driver_class.__name__, driver_type.value)
        return driver_class


class BosonicMoleculeDriver(BosonicDriver):
    """
    Molecule based bosonic driver
    """

    def __init__(
        self,
        molecule: Molecule,
        basis: str = "sto3g",
        driver_type: BosonicDriverType = BosonicDriverType.AUTO,
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

        Raises:
            MissingOptionalLibraryError: Driver not installed.
        """
        self._driver_class = BosonicDriverType.driver_class_from_type(driver_type)
        self._driver_kwargs = driver_kwargs
        super().__init__(basis=basis, supports_molecule=True)
        self.molecule = molecule

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
        driver = self._driver_class.from_molecule(self.molecule, self.basis, self.driver_kwargs)
        return driver.run()
