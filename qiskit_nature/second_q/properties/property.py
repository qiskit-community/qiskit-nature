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

from typing import TYPE_CHECKING

from abc import ABC
import logging
import sys

import h5py

from qiskit_nature.deprecation import warn_deprecated, DeprecatedType


if sys.version_info >= (3, 8):
    # pylint: disable=no-name-in-module
    from typing import runtime_checkable, Protocol
else:
    from typing_extensions import runtime_checkable, Protocol

if TYPE_CHECKING:
    from qiskit_nature.second_q.problems import EigenstateResult

LOGGER = logging.getLogger(__name__)


class Property(ABC):
    """The Property base class.

    A Property in Qiskit Nature provides the means to give meaning to a given set of raw data.
    As such, every Property is an object which constructs an operator to be evaluated during the
    problem solution and the interface provides the means for a user to write any custom Property
    (i.e. the user can evaluate custom *observables* by writing a class which can generate an
    operator out of a given set of raw data).
    """

    VERSION = 1
    """Each Property has its own version number. Although initially only defined on the base class,
    a subclass can increment its version number in order to handle changes during `load` and `save`
    operations."""

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
        logger.info(str(self))

    def to_hdf5(self, parent: h5py.Group) -> None:
        """Stores this instance in an HDF5 group inside of the provided parent group.

        See also :func:`~qiskit_nature.hdf5.HDF5Storable.to_hdf5` for more details.

        Args:
            parent: the parent HDF5 group.
        """
        group = parent.require_group(self.name)
        group.attrs["__class__"] = self.__class__.__name__
        group.attrs["__module__"] = self.__class__.__module__
        group.attrs["__version__"] = self.VERSION


class PseudoProperty(Property, ABC):
    """**DEPRECATED**: The PseudoProperty type.

    A pseudo-property is a type derived by auxiliary property-related meta data.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        warn_deprecated(
            "0.4.0",
            DeprecatedType.CLASS,
            "PseudoProperty",
            DeprecatedType.CLASS,
            "Interpretable",
            additional_msg=(
                "The PseudoProperty class is deprecated. Instead, of requiring an `interpret()` "
                "method on the Property base-class, this is now handled via the `Interpretable` "
                "protocol."
            ),
        )

    def interpret(self, result: "EigenstateResult") -> None:
        """A PseudoProperty cannot interpret anything."""
        pass


@runtime_checkable
class Interpretable(Protocol):
    """A protocol determining whether or not an object is interpretable.

    An object is considered interpretable if it implements an `interpret` method.
    """

    def interpret(self, result: "EigenstateResult") -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in the object's context.

        Args:
            result: the result to add meaning to.
        """
