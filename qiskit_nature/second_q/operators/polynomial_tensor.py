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
import numpy as np
from typing import Dict
from qiskit.opflow.mixins import StarAlgebraMixin
from qiskit.quantum_info.operators.mixins import TolerancesMixin


class PolynomialTensor(StarAlgebraMixin, TolerancesMixin):
    """Polynomial Tensor class"""

    def __init__(self, data: Dict[str, np.ndarray]):
        self._data = data

        for key, value in self._data.items():
            if len(np.shape(value)) == len(key):
                pass
            else:
                raise ValueError(
                    f"data key length {len(key)} and number of value dimensions {np.shape(value)} do not match"
                )

    def mul(self, other: complex):
        """scalar multiplication of PolynomialTensor with complex"""

        # other might be float, int or etc. quantum info - typing is Number (check)
        prod_dict = {}

        for key, matrix in self._data.items():
            prod_dict[key] = matrix * other
        return PolynomialTensor(prod_dict)

    def add(self, other: PolynomialTensor):
        """addition of PolynomialTensors"""

        # Polynomial tensor immutable!
        sum_dict = self._data  # or try empty dict {}

        if other is not isinstance(other, PolynomialTensor):
            raise TypeError("Incorrect argument type: other should be PolynomialTensor")

        for key, value in other.items():
            if key in self._data.keys():
                if np.shape(value) == np.shape(self._data[key]):
                    print(f"key {key} of same dimension present in both")
                    sum_dict[key] = np.add(value, self._data[key])
                else:
                    raise ValueError(
                        f"Dictionary value dimensions {np.shape(value)} and {np.shape(self._data[key])} do not match"
                    )
            else:
                print(f"adding a new key {key}")
                sum_dict[key] = value

        return PolynomialTensor(sum_dict)

    def compose(self, other):
        pass

    def adjoint(self):
        pass
