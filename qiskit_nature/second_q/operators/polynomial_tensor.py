# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Polynomial Tensor class"""

from __future__ import annotations
from typing import Dict
from numbers import Number
import numpy as np
from qiskit.quantum_info.operators.mixins import (
    LinearMixin,
    AdjointMixin,
    TolerancesMixin,
)


class PolynomialTensor(LinearMixin, AdjointMixin, TolerancesMixin):
    """Polynomial Tensor class"""

    def __init__(self, data: Dict[str, np.ndarray]):
        self._data = data

        for key, value in self._data.items():
            if len(np.shape(value)) == len(key):
                pass
            else:
                raise ValueError(
                    f"data key {key} of length {len(key)} does not match "
                    f"data value of dimensions {np.shape(value)}"
                )

    def mul(self, other: complex):
        """Scalar multiplication of PolynomialTensor with complex"""

        prod_dict = {}

        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        for key, matrix in self._data.items():
            prod_dict[key] = matrix * other
        return PolynomialTensor(prod_dict)

    def _multiply(self, other):
        return self.mul(other)

    def add(self, other: PolynomialTensor):
        """Addition of PolynomialTensors"""

        sum_dict = self._data.copy()

        if not isinstance(other, PolynomialTensor):
            raise TypeError("Incorrect argument type: other should be PolynomialTensor")

        for other_key, other_value in other._data.items():
            if other_key in sum_dict.keys():
                if np.shape(other_value) == np.shape(sum_dict[other_key]):
                    sum_dict[other_key] = np.add(other_value, sum_dict[other_key])
                else:
                    print("not same shape", np.shape(other_value), np.shape(sum_dict[other_key]))
                    raise ValueError(
                        f"For key {other_key} "
                        f"corresponding data value of shape {np.shape(sum_dict[other_key])} "
                        f"does not match value of shape {np.shape(other_value)}"
                    )
            else:
                sum_dict[other_key] = other_value

        return PolynomialTensor(sum_dict)

    def _add(self, other, qargs=None):
        return self.add(other)

    def __eq__(self, other):
        """Check equality of PolynomialTensors"""
        if self._data.keys() == other._data.keys():
            for key in self._data.keys():
                if np.allclose(self._data[key], other._data[key], atol=self.atol, rtol=self.rtol):
                    return isinstance(other, PolynomialTensor)
                else:
                    return False
        else:
            return False

    def conjugate(self):
        """Conjugate of PolynomialTensors"""
        conj_dict = {}

        for key, value in self._data.items():
            conj_dict[key] = np.conjugate(value)

        return PolynomialTensor(conj_dict)

    def transpose(self):
        """Transpose of PolynomialTensor"""

        transpose_dict = {}

        for key, value in self._data.items():
            transpose_dict[key] = np.transpose(value)

        return PolynomialTensor(transpose_dict)
