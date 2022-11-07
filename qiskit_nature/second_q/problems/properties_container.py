# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Property containment utilities."""

from __future__ import annotations

from collections.abc import MutableSet
from collections import OrderedDict
from typing import Generator

from qiskit_nature.second_q.properties import SparseLabelOpsFactory


class PropertiesContainer(MutableSet):
    """The properties container class.

    This class manages storage of :class:`~qiskit_nature.second_q.properties.Property` objects.
    These objects are operator factories, generating the `aux_operators` that will be passed down to
    quantum algorithms which support their evaluation.

    This container class can only contain **a single instance** of any
    :class:`~qiskit_nature.second_q.properties.Property` kind. It enforces this via a `MutableSet`
    implementation.
    """

    def __init__(self) -> None:
        self._properties: OrderedDict[str, SparseLabelOpsFactory] = OrderedDict()

    def add(self, value: SparseLabelOpsFactory) -> None:
        key = value.__class__.__name__
        if key in self._properties.keys():
            raise ValueError(
                f"An object of type '{key}' already exists in this Container. Please remove that "
                "first, if you want to replace it with this new instance."
            )
        self._properties[key] = value

    def discard(self, value: str | type | SparseLabelOpsFactory) -> None:
        key: str
        if isinstance(value, str):
            key = value
        elif isinstance(value, type):
            key = value.__name__
        elif isinstance(value, SparseLabelOpsFactory):
            key = value.__class__.__name__
        else:
            raise TypeError(
                "Only inputs of type 'str', 'type', or 'SparseLabelOpsFactory' are supported, not "
                f"{type(value)}."
            )

        self._properties.pop(key, None)

    def __contains__(self, key: object) -> bool:
        actual_key: str
        if isinstance(key, str):
            actual_key = key
        elif isinstance(key, type):
            actual_key = key.__name__
        elif isinstance(key, SparseLabelOpsFactory):
            actual_key = key.__class__.__name__
        else:
            raise TypeError(
                "Only inputs of type 'str', 'type', or 'SparseLabelOpsFactory' are supported, not "
                f"{type(key)}."
            )
        return actual_key in self._properties.keys()

    def __len__(self) -> int:
        return len(self._properties)

    def __iter__(self) -> Generator[SparseLabelOpsFactory, None, None]:
        for prop in self._properties.values():
            yield prop

    def _getter(self, _type: type) -> SparseLabelOpsFactory | None:
        """An internal utility method to handle the attribute getter implementation.

        Args:
            _type: the target SparseLabelOpsFactory type of this getter.

        Returns:
            The property of the corresponding type or `None` if it is not available.
        """
        return self._properties.get(_type.__name__, None)

    def _setter(self, _property: SparseLabelOpsFactory | None, _type: type) -> None:
        """An internal utility method to handle the attribute setter implementation.

        Args:
            _property: the SparseLabelOpsFactory to set. If `None`, the internally stored property of the
                indicated type (see next argument) will be discarded instead of set.
            _type: the target SparseLabelOpsFactory type of this setter.

        Raises:
            TypeError: if the provided SparseLabelOpsFactory does not match the indicate type.
        """
        if _property is None:
            self.discard(_type)
            return

        if not isinstance(_property, _type):
            raise TypeError(
                f"Only objects of type '{_type.__name__}' are supported, not {type(_property)}."
            )
        self.add(_property)
