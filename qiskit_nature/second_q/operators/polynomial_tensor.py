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
from typing import Dict, Iterator
from collections.abc import Mapping
from numbers import Number
import numpy as np
from qiskit.quantum_info.operators.mixins import (
    LinearMixin,
    AdjointMixin,
    TolerancesMixin,
)


class PolynomialTensor(LinearMixin, AdjointMixin, TolerancesMixin, Mapping):
    """Polynomial Tensor class"""

    def __init__(self, data: Mapping[str, np.ndarray | Number], register_length: int) -> None:
        """
        Args:
            data: coefficient container;
                  mapping of string-based operator keys to coefficient matrix values.
        Raises:
            ValueError: when length of operator key does not match dimensions of value matrix.
            ValueError: when value matrix does not have consistent dimensions.
            ValueError: when some or all value matrices in ``data`` have different dimensions.
        """
        copy_dict: Dict[str, np.ndarray] = {}

        shapes = set()
        for key, value in data.items():
            if isinstance(value, Number):
                value = np.asarray(value)

            if len(value.shape) != len(key):
                raise ValueError(
                    f"Data key {key} of length {len(key)} does not match "
                    f"data value matrix of dimensions {value.shape}"
                )

            dims = set(value.shape)

            if len(dims) > 1:
                raise ValueError(
                    f"For key {key}: dimensions of value matrix are not identical {value.shape}"
                )

            shapes.update(dims)
            copy_dict[key] = value

        if len(shapes) != 1:
            raise ValueError("Dimensions of value matrices in data dictionary are not identical.")

        self._data = copy_dict
        self._register_length = register_length

    @property
    def register_length(self) -> int:
        """ Returns register length of the operator key in `Polynomial Tensor` object """

        return self._register_length

    def __getitem__(self, __k: str) -> (np.ndarray | Number):
        """
        Returns value matrix in the `Polynomial Tensor` object.

        Args:
            __k: operator key string in the `Polynomial Tensor` object
        Returns:
            Value matrix corresponding to the operator key `__k`
        """

        return self._data.__getitem__(__k)

    def __len__(self) -> int:
        """
        Returns length of `Polynomial Tensor` object
        """

        return self._data.__len__()

    def __iter__(self) -> Iterator[str]:
        """
        Returns iterator of the `Polynomial Tensor` object
        """

        return self._data.__iter__()

    def _multiply(self, other: complex):
        """Scalar multiplication of PolynomialTensor with complex

        Args:
            other: scalar to be multiplied with the ``Polynomial Tensor`` object.
        Returns:
            the new ``Polynomial Tensor`` product object.
        Raises:
            TypeError: if ``other`` is not a ``Number``.
        """

        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        prod_dict: Dict[str, np.ndarray] = {}
        for key, matrix in self._data.items():
            prod_dict[key] = np.multiply(matrix, other)
        return PolynomialTensor(prod_dict, self._register_length)

    def _add(self, other: PolynomialTensor, qargs=None):
        """Addition of PolynomialTensors

        Args:
            other: second``Polynomial Tensor`` object to be added to the first.
        Returns:
            the new summed ``Polynomial Tensor`` object.
        Raises:
            TypeError: when ``other`` is not a ``Polynomial Tensor`` object.
            ValueError: when values corresponding to keys in ``other`` and
                            the first ``Polynomial Tensor`` object do not match.
        """

        if not isinstance(other, PolynomialTensor):
            raise TypeError("Incorrect argument type: other should be PolynomialTensor")

        sum_dict: Dict[str, np.ndarray] = self._data.copy()
        for other_key, other_value in other._data.items():
            if other_key in sum_dict.keys():
                if other_value.shape == np.shape(sum_dict[other_key]):
                    sum_dict[other_key] = np.add(other_value, sum_dict[other_key])
                else:
                    raise ValueError(
                        f"For key {other_key}: "
                        f"corresponding data value of shape {np.shape(sum_dict[other_key])} "
                        f"does not match other value matrix of shape {other_value.shape}"
                    )
            else:
                sum_dict[other_key] = other_value
        return PolynomialTensor(sum_dict, self._register_length)

    def __eq__(self, other):
        """Check equality of PolynomialTensors

        Args:
            other: second``Polynomial Tensor`` object to be compared with the first.
        Returns:
            True when ``Polynomial Tensor`` objects are equal, False when unequal.
        """

        if not isinstance(other, PolynomialTensor):
            return False

        if self._data.keys() != other._data.keys():
            return False
        for key, value in self._data.items():
            if not np.allclose(value, other._data[key], atol=self.atol, rtol=self.rtol):
                return False
        return True

    def conjugate(self) -> PolynomialTensor:
        """Conjugate of PolynomialTensors

        Returns:
            the complex conjugate of the ``Polynomial Tensor`` object.
        """

        conj_dict: Dict[str, np.ndarray] = {}
        for key, value in self._data.items():
            conj_dict[key] = np.conjugate(value)

        return PolynomialTensor(conj_dict, self._register_length)

    def transpose(self) -> PolynomialTensor:
        """Transpose of PolynomialTensor

        Returns:
            the transpose of the ``Polynomial Tensor`` object.
        """

        transpose_dict: Dict[str, np.ndarray] = {}
        for key, value in self._data.items():
            transpose_dict[key] = np.transpose(value)

        return PolynomialTensor(transpose_dict, self._register_length)
