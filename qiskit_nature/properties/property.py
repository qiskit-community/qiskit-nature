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

from abc import ABC, abstractmethod, abstractclassmethod
from typing import List, Union

from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.results import EigenstateResult


class Property(ABC):
    """The Property base class.

    A Property in Qiskit Nature provides the means to give meaning to a given set of raw data.
    As such, it provides the logic to transform a raw data (as e.g. produced by a
    `qiskit_nature.drivers.BaseDriver`) into a
    `qiskit_nature.operators.second_quantization.SecondQuantizedOp`.
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

    @abstractclassmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> "Property":
        """Construct a Property instance from a driver result.

        This method should implement the logic which is required to extract the raw data for a
        certain property from the result produced by a driver.

        Args:
            result: the driver result from which to extract the raw data.
        """

    @abstractmethod
    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Generates the (list of) second quantized operators associated with this Property."""

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an `qiskit_nature.result.EigenstateResult` in the context of this Property.

        This is currently a method stub which may be used in the future.

        Args:
            result: the result to add meaning to.
        """
        raise NotImplementedError()
