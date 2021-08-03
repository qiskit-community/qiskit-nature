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

"""The SecondQuantizedProperty base class."""

from abc import abstractmethod
from typing import Any, List, TypeVar, Union

from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization import SecondQuantizedOp

from ..grouped_property import GroupedProperty
from ..property import Property


class SecondQuantizedProperty(Property):
    """The SecondQuantizedProperty base class.

    A second-quantization property provides the logic to transform a raw data (as e.g. produced by a
    `qiskit_nature.second_quantization.drivers.BaseDriver`) into a
    `qiskit_nature.operators.second_quantization.SecondQuantizedOp`.
    """

    @abstractmethod
    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns the (list of) second quantized operators associated with this Property."""

    @classmethod
    @abstractmethod
    def from_legacy_driver_result(cls, result: Any) -> "Property":
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
    def _validate_input_type(cls, result: Any, valid_type: Any) -> None:
        # The type hint of `valid_type` is not easy to determine because we are passing a typing
        # alias which is a type hint itself. So what is the type hint for a type hint...
        # For the time being this should be fine because the logic around from_legacy_driver_result
        # will need to be slightly adapted *before* the next release anyways when we continue with
        # the integration of the `Property` objects.
        if not isinstance(result, valid_type.__args__):
            raise QiskitNatureError(
                f"You cannot construct an {cls.__name__} from a {result.__class__.__name__}. "
                "Please provide an object of any of these types instead: "
                f"{typ.__name__ for typ in valid_type.__args__}"
            )


# pylint: disable=invalid-name
T = TypeVar("T", bound=SecondQuantizedProperty, covariant=True)


class GroupedSecondQuantizedProperty(GroupedProperty[T], SecondQuantizedProperty):
    """A GroupedProperty subtype containing purely second-quantized properties."""

    @abstractmethod
    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """
        Returns the list of second quantized operators given by the properties contained in this
        group.
        """
