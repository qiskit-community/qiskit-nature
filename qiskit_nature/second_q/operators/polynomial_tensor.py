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
        if not isinstance(data, Dict):
            raise TypeError(f"data {data} must be a Dictionary")

        self._data = data

        shapes = ()
        for key, value in self._data.items():
            shapes += value.shape
            if len(value.shape) != len(key):
                raise ValueError(
                    f"data key {key} of length {len(key)} does not match "
                    f"data value matrix of dimensions {value.shape}"
                )
            if len(set(value.shape)) != 1:
                raise ValueError(
                    f"For key {key}: dimensions of value matrix are not identical {value.shape}"
                )

        if len(set(shapes)) != 1:
            raise ValueError("Dimensions of value matrices in data dictionary are not identical.")

    def mul(self, other: complex):
        """Scalar multiplication of PolynomialTensor with complex"""

        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        prod_dict = {}
        for key, matrix in self._data.items():
            prod_dict[key] = matrix * other
        return PolynomialTensor(prod_dict)

    def _multiply(self, other):
        return self.mul(other)

    def add(self, other: PolynomialTensor):
        """Addition of PolynomialTensors"""

        if not isinstance(other, PolynomialTensor):
            raise TypeError("Incorrect argument type: other should be PolynomialTensor")

        sum_dict = self._data.copy()
        for other_key, other_value in other._data.items():
            if other_key in sum_dict.keys():
                if other_value.shape == np.shape(sum_dict[other_key]):
                    sum_dict[other_key] = np.add(other_value, sum_dict[other_key])
                else:
                    print("not same shape", other_value.shape, np.shape(sum_dict[other_key]))
                    raise ValueError(
                        f"For key {other_key} "
                        f"corresponding data value of shape {np.shape(sum_dict[other_key])} "
                        f"does not match other value matrix of shape {other_value.shape}"
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
