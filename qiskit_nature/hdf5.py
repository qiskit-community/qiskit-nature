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

"""Qiskit Nature HDF5 Integration."""

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Generator

import h5py

from qiskit_nature import QiskitNatureError

if sys.version_info >= (3, 8):
    # pylint: disable=no-name-in-module
    from typing import runtime_checkable, Protocol
else:
    from typing_extensions import runtime_checkable, Protocol


LOGGER = logging.getLogger(__name__)


@runtime_checkable
class HDF5Storable(Protocol):
    """A Protocol implemented by those classes which support conversion methods for HDF5."""

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in a HDF5 group inside of the provided parent group.

        Qiskit Nature uses the convention of storing the `__module__` and `__class__` information as
        attributes of an HDF5 group. Furthermore, a `__version__` should be stored in order to allow
        version handling at runtime.

        Args:
            parent: the parent HDF5 group.
        """
        ...

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group) -> Any:
        """Constructs a new instance from the data stored in the provided HDF5 group.

        Args:
            h5py_group: the HDF5 group from which to load the data.

        Returns:
            A new instance of this class.
        """
        ...


def save_to_hdf5(obj: HDF5Storable, filename: str, force: bool = False) -> None:
    """A utility to method to store an object to an HDF5 file.

    Args:
        obj: the `HDF5Storable` object to store in the file.
        filename: the path to the HDF5 file.
        force: whether to forcefully overwrite an existing file.

    Raises:
        QiskitNatureError: if the file at the given path already exists and forcefully overwriting
            is not enabled.
    """
    if not isinstance(obj, HDF5Storable):
        LOGGER.error("%s is not an instance of %s", obj, HDF5Storable)
        return

    if Path(filename).exists() and not force:
        raise QiskitNatureError(
            f"The file at {filename} already exists! Specify `force=True` if you want to overwrite it!"
        )

    with h5py.File(filename, "w") as file:
        obj.to_hdf5(file)


def load_from_hdf5(filename: str) -> Generator[Any, None, None]:
    """Loads Qiskit Nature objects from an HDF5 file.

    Args:
        filename: the path to the HDF5 file.

    Yields:
        The objects constructed from the HDF5 groups encountered in the h5py_group.
    """
    with h5py.File(filename, "r") as file:
        yield from _import_and_build_from_hdf5(file)


def _import_and_build_from_hdf5(h5py_group: h5py.Group) -> Generator[Any, None, None]:
    """Imports and builds a Qiskit Nature object from an HDF5 group.

    Qiskit Nature uses the convention of storing the `__module__` and `__class__` information as
    attributes of an HDF5 group. From these, this method will import the required class at runtime
    and use its `form_hdf5` method to construct an instance of the encountered class.

    Args:
        h5py_group: the HDF5 group from which to import and build Qiskit Nature objects.

    Yields:
        The objects constructed from the HDF5 groups encountered in the h5py_group.
    """
    for group in h5py_group.values():
        module_path = group.attrs.get("__module__", "")
        if not module_path:
            continue

        class_name = group.attrs.get("__class__", "")

        if not class_name:
            LOGGER.warning("Skipping faulty object without a '__class__' attribute.")
            continue

        if not module_path.startswith("qiskit_nature"):
            LOGGER.warning("Skipping non-native object.")
            continue

        loaded_module = importlib.import_module(module_path)
        loaded_class = getattr(loaded_module, class_name, None)

        if loaded_class is None:
            LOGGER.warning(
                "Skipping object after failed import attempt of %s from %s",
                class_name,
                module_path,
            )
            continue

        if not issubclass(loaded_class, HDF5Storable):
            LOGGER.warning(
                "Skipping object because the loaded class %s is not a subclass of %s",
                loaded_class,
                HDF5Storable,
            )
            continue

        constructor = getattr(loaded_class, "from_hdf5")
        instance = constructor(group)
        yield instance
