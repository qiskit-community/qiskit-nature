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

"""Electronic property types."""

from typing import TypeVar

from .second_quantized_property import SecondQuantizedProperty, GroupedSecondQuantizedProperty


class ElectronicProperty(SecondQuantizedProperty):
    """The electronic Property type."""


# pylint: disable=invalid-name
T_co = TypeVar("T_co", bound=ElectronicProperty, covariant=True)


class GroupedElectronicProperty(GroupedSecondQuantizedProperty[T_co], ElectronicProperty):
    """A GroupedProperty subtype containing purely electronic properties."""
