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

"""PolynomialTensor class"""

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
    """PolynomialTensor class"""

    def __init__(self, data: Mapping[str, np.ndarray | Number], register_length: int) -> None:
        """
        Args:
            data: mapping of string-based operator keys to coefficient matrix values.
            register_length: dimensions of the value matrices in data mapping.
        Raises:
            ValueError: when length of operator key does not match dimensions of value matrix.
            ValueError: when value matrix does not have consistent dimensions.
            ValueError: when some or all value matrices in ``data`` have different dimensions.
        """
        copy_dict: Dict[str, np.ndarray] = {}

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
            if len(dims) == 1 and dims.pop() != register_length:
                raise ValueError(
                    f"Dimensions of value matrices in data dictionary do not match the provided "
                    f"register length, {register_length}"
                )

            copy_dict[key] = value

        self._data = copy_dict
        self._register_length = register_length

    @property
    def register_length(self) -> int:
        """Returns register length of the operator key in `PolynomialTensor`."""
        return self._register_length

    def __getitem__(self, __k: str) -> (np.ndarray | Number):
        """
        Returns value matrix in the `PolynomialTensor`.

        Args:
            __k: operator key string in the `PolynomialTensor`.
        Returns:
            Value matrix corresponding to the operator key `__k`
        """
        return self._data.__getitem__(__k)

    def __len__(self) -> int:
        """
        Returns length of `PolynomialTensor`.
        """
        return self._data.__len__()

    def __iter__(self) -> Iterator[str]:
        """
        Returns iterator of the `PolynomialTensor`.
        """
        return self._data.__iter__()

    def _multiply(self, other: complex) -> PolynomialTensor:
        """Scalar multiplication of PolynomialTensor with complex

        Args:
            other: scalar to be multiplied with the ``PolynomialTensor``.
        Returns:
            the new ``PolynomialTensor`` product object.
        Raises:
            TypeError: if ``other`` is not a ``Number``.
        """
        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        prod_dict: Dict[str, np.ndarray] = {}
        for key, matrix in self._data.items():
            prod_dict[key] = np.multiply(matrix, other)
        return PolynomialTensor(prod_dict, self._register_length)

    def _add(self, other: PolynomialTensor, qargs=None) -> PolynomialTensor:
        """Addition of PolynomialTensors

        Args:
            other: second``PolynomialTensor`` object to be added to the first.
        Returns:
            the new summed ``PolynomialTensor``.
        Raises:
            TypeError: when ``other`` is not a ``PolynomialTensor``.
            ValueError: when values corresponding to keys in ``other`` and
                            the first ``PolynomialTensor`` object do not match.
        """
        if not isinstance(other, PolynomialTensor):
            raise TypeError("Incorrect argument type: other should be PolynomialTensor")

        if self.register_length != other.register_length:
            raise ValueError(
                "The dimensions of the PolynomialTensors which are to be added together, do not "
                f"match: {self.register_length} != {other.register_length}"
            )

        sum_dict = {key: np.value + other._data.get(key, 0) for key, value in self._data.items()}
        other_unique = {key: other._data[key] for key in other._data.keys() - self._data.keys()}
        sum_dict.update(other_unique)
        return PolynomialTensor(sum_dict, self._register_length)

    def __eq__(self, other: object) -> bool:
        """Check equality of first PolynomialTensor with other

        Args:
            other: second``PolynomialTensor`` object to be compared with the first.
        Returns:
            True when ``PolynomialTensor`` objects are equal, False when unequal.
        """
        if not isinstance(other, PolynomialTensor):
            return False

        if self._register_length != other._register_length:
            return False

        if self._data.keys() != other._data.keys():
            return False

        for key, value in self._data.items():
            if not np.array_equal(value, other._data[key]):
                return False
        return True

    def equiv(self, other: object) -> bool:
        """Check equivalence of first PolynomialTensor with other

        Args:
            other: second``PolynomialTensor`` object to be compared with the first.
        Returns:
            True when ``PolynomialTensor`` objects are equivalent, False when not.
        """
        if not isinstance(other, PolynomialTensor):
            return False

        if self._register_length != other._register_length:
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
            the complex conjugate of the ``PolynomialTensor``.
        """
        conj_dict: Dict[str, np.ndarray] = {}
        for key, value in self._data.items():
            conj_dict[key] = np.conjugate(value)

        return PolynomialTensor(conj_dict, self._register_length)

    def transpose(self) -> PolynomialTensor:
        """Transpose of PolynomialTensor

        Returns:
            the transpose of the ``PolynomialTensor``.
        """
        transpose_dict: Dict[str, np.ndarray] = {}
        for key, value in self._data.items():
            transpose_dict[key] = np.transpose(value)

        return PolynomialTensor(transpose_dict, self._register_length)
