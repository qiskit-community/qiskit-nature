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

from qiskit_nature.second_q.properties import Property


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
        self._properties: OrderedDict[str, Property] = OrderedDict()

    def add(self, value: Property) -> None:
        key = value.__class__.__name__
        if key in self._properties.keys():
            raise ValueError(
                f"An object of type '{key}' already exists in this Container. Please remove that "
                "first, if you want to replace it with this new instance."
            )
        self._properties[key] = value

    def discard(self, value: str | type | Property) -> None:
        key: str
        if isinstance(value, str):
            key = value
        elif isinstance(value, type):
            key = value.__name__
        elif isinstance(value, Property):
            key = value.__class__.__name__
        else:
            raise TypeError(
                f"Only inputs of type 'str', 'type', or 'Property' are supported, not {type(value)}."
            )

        self._properties.pop(key, None)

    def __contains__(self, key: object) -> bool:
        actual_key: str
        if isinstance(key, str):
            actual_key = key
        elif isinstance(key, type):
            actual_key = key.__name__
        elif isinstance(key, Property):
            actual_key = key.__class__.__name__
        else:
            raise TypeError(
                f"Only inputs of type 'str', 'type', or 'Property' are supported, not {type(key)}."
            )
        return actual_key in self._properties.keys()

    def __len__(self) -> int:
        return len(self._properties)

    def __iter__(self) -> Generator[Property, None, None]:
        for prop in self._properties.values():
            yield prop
