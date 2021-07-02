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

"""A Composite of multiple properties."""

from collections.abc import Iterable
from typing import Dict, Optional, Type, Union

from .property import Property


class CompositeProperty(Property, Iterable):
    """A Composite of multiple properties."""

    def __init__(self, name: str) -> None:
        """
        Args:
            name: the name of the property.
        """
        super().__init__(name)
        self._properties: Dict[str, Property] = {}

    def add_property(self, property: Property) -> None:
        """Adds a property to the composite.

        Args:
            property: the property to be added.
        """
        self._properties[property.name] = property

    def get_property(
        self, prop: Union[str, Type[Property]]
    ) -> Optional[Property]:
        """TODO."""
        name: str
        if isinstance(prop, str):
            name = prop
        else:
            name = prop.__name__
        return self._properties.get(name, None)

    def __iter__(self):
        """Returns the iterator method.

        This method should be a generator function.
        """
        return self.generator()

    def generator(self):
        """The generator function iterating over all internal properties."""
        for property in self._properties.values():
            new_property = (yield property)
            if new_property is not None:
                self.add_property(new_property)
