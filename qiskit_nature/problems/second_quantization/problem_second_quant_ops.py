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
"""Problem Second Quantized Operators"""
from typing import List

from qiskit_nature.operators.second_quantization import SecondQuantizedOp


class ProblemSecondQuantOps():
    """Problem Second Quantized Operators"""

    def __init__(self, main_operator: SecondQuantizedOp,
                 aux_operators_list: List[SecondQuantizedOp] = None):
        """

        Args:
            main_operator: main problem-related second quantized operator.
            aux_operators_list: a list of auxiliary problem-related second quantized operators.
        """

        self._main_operator = main_operator
        self._aux_operators_list = aux_operators_list

    def __len__(self):
        length = 0
        if self._main_operator is not None:
            length += 1
        if self._aux_operators_list:
            length += len(self._aux_operators_list)
        return length

    def __iter__(self):
        for operator in [self._main_operator] + self._aux_operators_list:
            yield operator

    @property
    def main_operator(self) -> SecondQuantizedOp:
        """Returns the main problem-related second quantized operator."""
        return self._main_operator

    @property
    def aux_operators_list(self) -> List[SecondQuantizedOp]:
        """Returns a list of auxiliary problem-related second quantized operators."""
        return self._aux_operators_list
