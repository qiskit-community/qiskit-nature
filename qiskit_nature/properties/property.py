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

"""The Property base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import logging

import h5py

from qiskit_nature import __version__
from qiskit_nature.results import EigenstateResult

LOGGER = logging.getLogger(__name__)


class Property(ABC):
    """The Property base class.

    A Property in Qiskit Nature provides the means to give meaning to a given set of raw data.
    As such, every Property is an object which constructs an operator to be evaluated during the
    problem solution and the interface provides the means for a user to write any custom Property
    (i.e. the user can evaluate custom *observables* by writing a class which can generate an
    operator out of a given set of raw data).
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name: the name of the property.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Sets the name."""
        self._name = name

    def __str__(self) -> str:
        return self.name

    def log(self) -> None:
        """Logs the Property information."""
        logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        if not logger.isEnabledFor(logging.INFO):
            return
        logger.info(self.__str__())

    @abstractmethod
    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

        Args:
            result: the result to add meaning to.
        """
        raise NotImplementedError()

    def save(self, filename: str, append: bool = False) -> None:
        """TODO."""
        with h5py.File(filename, "a" if append else "w") as file:
            if not append:
                file.attrs["__version__"] = __version__
            self.to_hdf5(file)

    def to_hdf5(self, parent: h5py.Group) -> None:
        """TODO."""
        group = parent.require_group(self.name)
        group.attrs["__class__"] = self.__class__.__name__
        group.attrs["__module__"] = self.__class__.__module__

    @staticmethod
    def load(filename):
        """TODO."""
        with h5py.File(filename, "r") as file:
            if file.attrs["__version__"] != __version__:
                LOGGER.warning(
                    "This HDF5 was written with Qiskit Nature version %s but you are using version "
                    "%s.",
                    file.attrs["__version__"],
                    __version__,
                )
            yield from Property.import_and_build_from_hdf5(file)

    @staticmethod
    def import_and_build_from_hdf5(h5py_group: h5py.Group):
        """TODO."""
        for group in h5py_group.values():
            module_path = group.attrs.get("__module__", "")
            class_name = group.attrs.get("__class__", "")

            if not class_name:
                LOGGER.warning("Skipping faulty object without a '__class__' attribute.")
                continue

            if not module_path.startswith("qiskit_nature.properties") and not (
                module_path == "qiskit_nature.drivers.molecule" and class_name == "Molecule"
            ):
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

            constructor = getattr(loaded_class, "from_hdf5")
            instance = constructor(group)
            yield instance

    @classmethod
    def from_hdf5(cls, h5py_group: h5py.Group):
        """TODO."""
        # TODO: un-comment once all sub-classes actually implement this
        # raise NotImplementedError()


class PseudoProperty(Property, ABC):
    """The PseudoProperty type.

    A pseudo-property is a type derived by auxiliary property-related meta data.
    """

    def interpret(self, result: EigenstateResult) -> None:
        """A PseudoProperty cannot interpret anything."""
        pass
