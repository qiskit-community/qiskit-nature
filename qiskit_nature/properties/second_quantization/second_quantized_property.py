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

"""The SecondQuantizedProperty base class."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Type, TypeVar, Union

from qiskit_nature import ListOrDictType, QiskitNatureError
from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import SecondQuantizedOp

from ..grouped_property import GroupedProperty
from ..property import Property

LegacyDriverResult = Union[QMolecule, WatsonHamiltonian]


class SecondQuantizedProperty(Property):
    """The SecondQuantizedProperty base class.

    A second-quantization property provides the logic to transform the raw data placed into it by
    e.g. a :class:`qiskit_nature.drivers.second_quantization.BaseDriver` into a
    :class:`qiskit_nature.operators.second_quantization.SecondQuantizedOp`.
    """

    @abstractmethod
    def second_q_ops(self) -> ListOrDictType[SecondQuantizedOp]:
        """Returns the second quantized operators associated with this Property.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `SecondQuantizedOp` objects.
        """

    @classmethod
    @abstractmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> Property:
        """Construct a :class:`~qiskit_nature.properties.Property` instance from a legacy driver
        result.

        This method should implement the logic which is required to extract the raw data for a
        certain property from the result produced by a legacy driver.

        Args:
            result: the legacy driver result from which to extract the raw data.

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if an invalid legacy driver result type is passed.
        """

    @classmethod
    def _validate_input_type(cls, result: Any, valid_type: Type) -> None:
        if not isinstance(result, valid_type):
            raise QiskitNatureError(
                f"You cannot construct an {cls.__name__} from a {result.__class__.__name__}. "
                f"Please provide an object of type {valid_type} instead."
            )


# pylint: disable=invalid-name
T = TypeVar("T", bound=SecondQuantizedProperty, covariant=True)


class GroupedSecondQuantizedProperty(GroupedProperty[T], SecondQuantizedProperty):
    """A :class:`~qiskit_nature.properties.GroupedProperty` subtype containing purely
    second-quantized properties."""

    @abstractmethod
    def second_q_ops(self) -> ListOrDictType[SecondQuantizedOp]:
        """Returns the second quantized operators associated with the properties in this group.

        The actual return-type is determined by `qiskit_nature.settings.dict_aux_operators`.

        Returns:
            A `list` or `dict` of `SecondQuantizedOp` objects.
        """
