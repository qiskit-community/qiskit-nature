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

import importlib
from collections.abc import Iterable
from typing import Dict, Generator, Generic, Optional, Type, TypeVar, Union

import h5py

from qiskit_nature.results import EigenstateResult
from .property import Property, PseudoProperty

# pylint: disable=invalid-name
T = TypeVar("T", bound=Property, covariant=True)


class GroupedProperty(Property, Iterable, Generic[T]):
    """A group of multiple properties.

    This class implements the Composite Pattern [1]. As such, it acts as both, a container of
    multiple :class:`~qiskit_nature.properties.Property` instances as well as a
    :class:`~qiskit_nature.properties.Property` itself.  :class:`~qiskit_nature.properties.Property`
    objects can be added and accessed via the ``add_property`` and ``get_property`` methods,
    respectively.

    The internal data container stores :class:`~qiskit_nature.properties.Property` objects by name.
    This has the side effect that each object stored in this group must have a unique name.

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

        :class:`~qiskit_nature.properties.property.PseudoProperty` objects are automatically
        excluded.

        [1]: https://docs.python.org/3/reference/expressions.html#generator-iterator-methods
        """
        for prop in self._properties.values():
            if isinstance(prop, PseudoProperty):
                continue
            new_property = yield prop
            if new_property is not None:
                self.add_property(new_property)

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

        Args:
            result: the result to add meaning to.
        """
        for prop in self._properties.values():
            prop.interpret(result)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """TODO."""
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        for prop in self._properties.values():
            prop.to_hdf5(group)

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group) -> "GroupedProperty":
        """TODO."""
        module_path = h5py_group.attrs["__module__"]
        class_name = h5py_group.attrs["__class__"]
        # TODO: use `.get(..., None)` and handle missing values

        loaded_module = importlib.import_module(module_path)
        loaded_class = getattr(loaded_module, class_name, None)
        # TODO: handle missing loaded_class

        # NOTE: the following relies on an initializer which does _NOT_ take any arguments!
        # Currently, this is not the case in our design. However, I think it should be (see comments
        # in respective sub-classes).
        ret = loaded_class()

        for prop in Property.import_and_build_from_hdf5(h5py_group):
            ret.add_property(prop)

        return ret
