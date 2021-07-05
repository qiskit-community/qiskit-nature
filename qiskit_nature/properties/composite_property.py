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
from typing import Dict, Generator, Generic, Optional, Type, TypeVar, Union

from qiskit_nature.results import EigenstateResult
from .property import Property

# pylint: disable=invalid-name
T = TypeVar("T", bound=Property)


class CompositeProperty(Property, Iterable, Generic[T]):
    """A Composite of multiple properties."""

    def __init__(self, name: str) -> None:
        """
        Args:
            name: the name of the property.
        """
        super().__init__(name)
        self._properties: Dict[str, T] = {}

    def __repr__(self) -> str:
        string = super().__repr__() + ":"
        for prop in self._properties.values():
            for line in str(prop).split("\n"):
                string += f"\n\t{line}"
        return string

    def add_property(self, prop: Optional[T]) -> None:
        """Adds a property to the composite.

        Args:
            prop: the property to be added.
        """
        if prop is not None:
            self._properties[prop.name] = prop

    def get_property(self, prop: Union[str, Type[Property]]) -> Optional[T]:
        """Gets a property from the Composite.

        Args:
            prop: the name or type of the property to get from the Composite.

        Returns:
            The queried property (or None).
        """
        name: str
        if isinstance(prop, str):
            name = prop
        else:
            name = prop.__name__
        return self._properties.get(name, None)

    def __iter__(self) -> Generator[T, T, None]:
        """Returns the generator-iterator method."""
        return self._generator()

    def _generator(self) -> Generator[T, T, None]:
        """A generator-iterator method [1] iterating over all internal properties.

        [1]: https://docs.python.org/3/reference/expressions.html#generator-iterator-methods
        """
        for prop in self._properties.values():
            new_property = yield prop
            if new_property is not None:
                self.add_property(new_property)

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:~qiskit_nature.result.EigenstateResult in this property's context.

        Args:
            result: the result to add meaning to.
        """
        for prop in self._properties.values():
            prop.interpret(result)
