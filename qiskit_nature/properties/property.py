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
from typing import Any, List, Union

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers import QMolecule as LegacyQMolecule
from qiskit_nature.drivers import WatsonHamiltonian as LegacyWatsonHamiltonian
from qiskit_nature.drivers.second_quantization import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.results import EigenstateResult

ElectronicDriverResult = Union[QMolecule, LegacyQMolecule]
VibrationalDriverResult = Union[WatsonHamiltonian, LegacyWatsonHamiltonian]
DriverResult = Union[ElectronicDriverResult, VibrationalDriverResult]


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
    def from_driver_result(cls, result: DriverResult) -> "Property":
        """Construct a Property instance from a driver result.

        This method should implement the logic which is required to extract the raw data for a
        certain property from the result produced by a driver.

        Args:
            result: the driver result from which to extract the raw data.

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if an invalid driver result type is passed.
        """

    @classmethod
    def _validate_input_type(cls, result: DriverResult, valid_type: Any) -> None:
        # The type hint of `valid_type` is not easy to determine because we are passing a typing
        # alias which is a type hint itself. So what is the type hint for a type hint...
        # For the time being this should be fine because the logic around `from_driver_result` will
        # need to slightly adapted *before* the next release anyways when we continue with the
        # integration of the `Property` objects.
        if not isinstance(result, valid_type.__args__):
            raise QiskitNatureError(
                f"You cannot construct an {cls.__name__} from a {result.__class__.__name__}. "
                "Please provide an object of any of these types instead: "
                f"{typ.__name__ for typ in valid_type.__args__}"
            )

    @abstractmethod
    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns the (list of) second quantized operators associated with this Property."""

    # TODO: use this to replace the result interpreter utilities of the structure problems?
    # This requires that a property-gathering super-object (e.g. ElectronicDriverResult) exists
    # which is stored inside of the ElectronicStructureProblem instead of the currently stored
    # QMolecule (vibrational case accordingly).
    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an `qiskit_nature.result.EigenstateResult` in the context of this Property.

        This is currently a method stub which may be used in the future.

        Args:
            result: the result to add meaning to.
        """
        raise NotImplementedError()
