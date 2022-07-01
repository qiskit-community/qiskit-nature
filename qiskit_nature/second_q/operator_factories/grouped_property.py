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

"""A group of multiple properties."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generator, Generic, Optional, Type, TypeVar, Union

import h5py
from qiskit_nature.hdf5 import _import_and_build_from_hdf5

from .property import Interpretable, Property

# pylint: disable=invalid-name
T_co = TypeVar("T_co", bound=Property, covariant=True)


class GroupedProperty(Property, Iterable, Generic[T_co]):
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
        self._properties: dict[str, T_co] = {}

    def __str__(self) -> str:
        string = [super().__str__() + ":"]
        for prop in self._properties.values():
            for line in str(prop).split("\n"):
                string += [f"\t{line}"]
        return "\n".join(string)

    def add_property(self, prop: Optional[T_co]) -> None:
        """Adds a property to the group.

        Args:
            prop: the property to be added.
        """
        if prop is not None:
            try:
                name = prop.name
            except AttributeError:
                name = prop.__class__.__name__
            self._properties[name] = prop

    def get_property(self, prop: Union[str, Type[Property]]) -> Optional[T_co]:
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

    def __iter__(self) -> Generator[T_co, T_co, None]:
        """Returns the generator-iterator method."""
        return self._generator()

    def _generator(self) -> Generator[T_co, T_co, None]:
        """A generator-iterator method [1] iterating over all internal properties.

        [1]: https://docs.python.org/3/reference/expressions.html#generator-iterator-methods
        """
        for prop in self._properties.values():
            new_property = yield prop
            if new_property is not None:
                self.add_property(new_property)

    def interpret(self, result: str) -> None:
        """Interprets an :class:`~qiskit_nature.results.str` in this property's context.

        Args:
            result: the result to add meaning to.
        """
        for prop in self._properties.values():
            if isinstance(prop, Interpretable):
                prop.interpret(result)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        This method also iterates all properties contained in this ``GroupProperty`` instance.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        super().to_hdf5(parent)
        group = parent.require_group(self.name)

        for prop in self._properties.values():
            prop.to_hdf5(group)

    @staticmethod
    def from_hdf5(h5py_group: h5py.Group) -> GroupedProperty:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        More specifically this method will iterate all groups found within `h5py_group` and
        constructs the corresponding objects from these groups.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.from_hdf5` for more details.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        class_name = h5py_group.attrs.get("__class__", "")

        ret: GroupedProperty = GroupedProperty(class_name)

        for prop in _import_and_build_from_hdf5(h5py_group):
            ret.add_property(prop)

        return ret
