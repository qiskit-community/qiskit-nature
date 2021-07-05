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

"""The Property base class."""

from abc import ABC, abstractmethod

from qiskit_nature.results import EigenstateResult


class Property(ABC):
    """The Property base class.

    A Property in Qiskit Nature provides the means to give meaning to a given set of raw data.
    As such, every Property is an object which constructs an operator to be evaluated during the
    problem solution and the interface provides the means for a user to write any custom Property
    (i.e. the user can evaluate custom _observables_ by writing a class which can generate an
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

    def __repr__(self) -> str:
        return self.name

    @abstractmethod
    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:~qiskit_nature.result.EigenstateResult in this property's context.

        Args:
            result: the result to add meaning to.
        """
        raise NotImplementedError()
