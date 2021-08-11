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

"""A group of multiple properties."""

from collections.abc import Iterable
from typing import Dict, Generator, Generic, Optional, Type, TypeVar, Union

from qiskit_nature.results import EigenstateResult
from .property import Property, PseudoProperty

# pylint: disable=invalid-name
T = TypeVar("T", bound=Property, covariant=True)


class GroupedProperty(Property, Iterable, Generic[T]):
    """A group of multiple properties.

    This class implements the Composite Pattern [1]. As such, it acts as both, a container of
    multiple Property instances as well as a Property itself.
    Property objects can be added and accessed via the `add_property` and `get_property` methods,
    respectively.

    The internal data container stores Property objects by name. This has the side effect that each
    object stored in this group must have a unique name.

    [1]: https://en.wikipedia.org/wiki/Composite_pattern
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name: the name of the property group.
        """
        super().__init__(name)
        self._properties: Dict[str, T] = {}

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        for prop in self._properties.values():
            for line in str(prop).split("\n"):
                string += [f"\t{line}"]
        return "\n".join(string)

    def add_property(self, prop: Optional[T]) -> None:
        """Adds a property to the group.

        Args:
            prop: the property to be added.
        """
        if prop is not None:
            self._properties[prop.name] = prop

    def get_property(self, prop: Union[str, Type[Property]]) -> Optional[T]:
        """Gets a property from the group.

        Args:
            prop: the name or type of the property to get from the group.

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

        `PseudoProperty` objects are automatically excluded.

        [1]: https://docs.python.org/3/reference/expressions.html#generator-iterator-methods
        """
        for prop in self._properties.values():
            if isinstance(prop, PseudoProperty):
                continue
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
