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

"""TODO."""

import importlib
import logging
import sys
from typing import Any, Generator

import h5py

if sys.version_info >= (3, 8):
    # pylint: disable=no-name-in-module
    from typing import runtime_checkable, Protocol
else:
    from typing_extensions import runtime_checkable, Protocol


LOGGER = logging.getLogger(__name__)


@runtime_checkable
class HDF5Storable(Protocol):
    """TODO."""

    def to_hdf5(self, parent: h5py.Group) -> None:
        """TODO."""
        ...

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group) -> Any:
        """TODO."""
        ...


def save_to_hdf5(obj: HDF5Storable, filename: str, append: bool = False) -> None:
    """TODO."""
    if not isinstance(obj, HDF5Storable):
        LOGGER.error("%s is not an instance of %s", obj, HDF5Storable)
        return
    with h5py.File(filename, "a" if append else "w") as file:
        obj.to_hdf5(file)


def load_from_hdf5(filename: str) -> Generator[Any, None, None]:
    """TODO."""
    with h5py.File(filename, "r") as file:
        yield from import_and_build_from_hdf5(file)


def import_and_build_from_hdf5(h5py_group: h5py.Group) -> Generator[Any, None, None]:
    """TODO."""
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
