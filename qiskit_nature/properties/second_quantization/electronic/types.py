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

"""Electronic property types."""

from typing import Optional, TypeVar

from qiskit_nature import QiskitNatureError
from ...property import PseudoProperty
from ..second_quantized_property import SecondQuantizedProperty, GroupedSecondQuantizedProperty


class ElectronicProperty(SecondQuantizedProperty):
    """The electronic Property type."""


# pylint: disable=invalid-name
T = TypeVar("T", bound=ElectronicProperty, covariant=True)


class GroupedElectronicProperty(GroupedSecondQuantizedProperty[T], ElectronicProperty):
    """A GroupedProperty subtype containing purely electronic properties."""

    def add_property(self, prop: Optional[T]) -> None:
        """Adds a property to the group.

        Args:
            prop: the property to be added.

        Raises:
            QiskitNatureError: if the added property is not an electronic one.
        """
        if prop is not None:
            if not isinstance(prop, (ElectronicProperty, PseudoProperty)):
                raise QiskitNatureError(
                    f"{prop.__class__.__name__} is not an instance of `ElectronicProperty`, which "
                    "it must be in order to be added to an `GroupedElectronicProperty`!"
                )
            self._properties[prop.name] = prop
