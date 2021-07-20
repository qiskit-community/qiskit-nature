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

"""The DriverResult class."""

from typing import TypeVar

from ..composite_property import CompositeProperty
from .second_quantized_property import SecondQuantizedProperty

# pylint: disable=invalid-name
T = TypeVar("T", bound=SecondQuantizedProperty)


class DriverResult(CompositeProperty[T], SecondQuantizedProperty):
    """The CompositeProperty result produced by a second quantization driver."""


class DriverMetadata:
    """A meta-data storage container for driver information."""

    def __init__(self, name: str, version: str, config: str) -> None:
        """
        Args:
            name: the name of the classical code run by the driver.
            version: the version of the classical code.
            config: the configuration of the classical code.
        """
        self.name = name
        self.version = version
        self.config = config
