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
from typing import Any, Type

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators import SecondQuantizedOp

from .property import Property


class SecondQuantizedProperty(Property):
    """The SecondQuantizedProperty base class.

    A second-quantization property provides the logic to transform the raw data placed into it by
    e.g. a :class:`qiskit_nature.second_q.drivers.BaseDriver` into a
    :class:`qiskit_nature.second_q.operators.SecondQuantizedOp`.
    """

    @abstractmethod
    def second_q_ops(self) -> dict[str, SecondQuantizedOp]:
        """Returns the second quantized operators associated with this Property.

        Returns:
            A `dict` of `SecondQuantizedOp` objects.
        """

    @classmethod
    def _validate_input_type(cls, result: Any, valid_type: Type) -> None:
        if not isinstance(result, valid_type):
            raise QiskitNatureError(
                f"You cannot construct an {cls.__name__} from a {result.__class__.__name__}. "
                f"Please provide an object of type {valid_type} instead."
            )
