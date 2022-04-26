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

"""A fix to avoid running the driver when retrieving second quantized operators."""

from typing import Optional
import logging

from qiskit_nature import ListOrDictType
from qiskit_nature.operators.second_quantization import SecondQuantizedOp

from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.properties.second_quantization import GroupedSecondQuantizedProperty

logger = logging.getLogger(__name__)


class CustomProblem(ElectronicStructureProblem):
    """A fix to avoid running the driver when retrieving second quantized operators."""

    def second_q_ops(self) -> ListOrDictType[SecondQuantizedOp]:

        if self._grouped_property is None:
            driver_result = self.driver.run()
            self._grouped_property = driver_result

        self._grouped_property_transformed = self._transform(self._grouped_property)
        second_quantized_ops = self._grouped_property_transformed.second_q_ops()

        return second_quantized_ops

    @property
    def grouped_property_transformed(self) -> Optional[GroupedSecondQuantizedProperty]:
        return self._grouped_property_transformed

    @grouped_property_transformed.setter
    def grouped_property_transformed(self, gpt):
        self._grouped_property_transformed = gpt
