# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Second-Quantized Operator Transformation interface."""

from abc import ABC, abstractmethod

from qiskit.chemistry.operators import SecondQuantizedOperator


class SecondQuantizedTransformation(ABC):
    """The interface for implementing methods which map from one `SecondQuantizedOperator` to
    another. These methods may or may not affect the size of the Hilbert space underlying the
    operator.
    """
    # TODO Do all of the transformations that we will come up with have some common side effect
    # which we need to account for? E.g., both, the active-space as well as the particle-hole
    # transformation, result in an energy offset which needs to be accounted for in the problem's
    # total energy. However, the seniority-zero transformation does not produce such an energy
    # offset which is why this example cannot be taken into the interface unless we default the
    # produced energy offset to 0.

    @abstractmethod
    def transform(self, second_q_op: SecondQuantizedOperator) -> SecondQuantizedOperator:
        """Transforms one `SecondQuantizedOperator` into another one. This may or may not affect the
        size of the Hilbert space underlying the operator.

        Args:
            second_q_op: the `SecondQuantizedOperator` to be transformed.

        Returns:
            A new `SecondQuantizedOperator` instance.
        """
        raise NotImplementedError()
