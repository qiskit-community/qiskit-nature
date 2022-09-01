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

"""The Sparse Label Operator base class."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Iterator
from numbers import Number
import cmath
import numpy as np

from qiskit.quantum_info.operators.mixins import LinearMixin, AdjointMixin, TolerancesMixin


class SparseLabelOp(LinearMixin, AdjointMixin, TolerancesMixin, ABC, Mapping):
    """The Sparse Label Operator base class."""

    def __init__(self, data: Mapping[str, complex], register_length: int, *, copy: bool = True):
        """
        Args:
            data: the operator data, mapping string-based keys to numerical values.
            register_length: the length of the operators register. This coincides with the maximum
                index on which an operation may be performed by this operator.
            copy: when set to False the `data` will not be copied and the dictionary will be
                stored by reference rather than by value (which is the default). Note, that this
                might requires you to not change the contents of the dictionary after constructing
                the operator. Use with care!
        """
        self._data: Mapping[str, complex] = {}
        if copy:
            self._data = dict(data.items())
        else:
            self._data = data
        self._register_length = register_length

    @property
    def register_length(self) -> int:
        """Returns the register length"""
        return self._register_length

    def _add(self, other: SparseLabelOp, qargs=None) -> SparseLabelOp:
        """Return Operator addition of self and other.

        Args:
            other: the second ``SparseLabelOp`` to add to the first.

        Returns:
            the new summed ``SparseLabelOp``.

        Raises:
            ValueError: when ``qargs`` argument is not ``None``
        """
        if not isinstance(other, SparseLabelOp):
            raise ValueError(
                f"Unsupported operand type(s) for +: 'SparseLabelOp' and '{type(other).__name__}'"
            )

        new_data = {key: value + other._data.get(key, 0) for key, value in self._data.items()}
        other_unique = {key: other._data[key] for key in other._data.keys() - self._data.keys()}
        new_data.update(other_unique)

        register_length = max(self.register_length, other.register_length)

        return self.__class__(new_data, register_length, copy=False)

    def _multiply(self, other: complex) -> SparseLabelOp:
        """Return scalar multiplication of self and other.

        Args:
            other: the number to multiply the ``SparseLabelOp`` values by.

        Returns:
            the newly multiplied ``SparseLabelOp``.

        Raises:
            TypeError: if ``other`` is not compatible type (int, float or complex)
        """
        if not isinstance(other, Number):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SparseLabelOp' and '{type(other).__name__}'"
            )
        new_data = {key: val * other for key, val in self._data.items()}

        return self.__class__(new_data, self.register_length, copy=False)

    def conjugate(self) -> SparseLabelOp:
        """Returns the conjugate of the ``SparseLabelOp``.

        Returns:
            the complex conjugate of the starting ``SparseLabelOp``.
        """
        new_data = {key: np.conjugate(val) for key, val in self._data.items()}

        return self.__class__(new_data, self.register_length, copy=False)

    @abstractmethod
    def transpose(self) -> SparseLabelOp:
        """Returns the transpose of the ``SparseLabelOp``.

        Returns:
            the transpose of the starting ``SparseLabelOp``.
        """

    def equiv(self, other: SparseLabelOp) -> bool:
        """Check equivalence of two ``SparseLabelOp`` instances up to an accepted tolerance.

        The absolute and relative tolerances can be changed via the `atol` and `rtol` attributes,
        respectively.

        Args:
            other: the second ``SparseLabelOp`` to compare with this instance.

        Returns:
            True if operators are equivalent, False if not.
        """
        if not isinstance(other, SparseLabelOp):
            return False
        if self.register_length != other._register_length:
            return False
        if self._data.keys() != other._data.keys():
            return False
        for key, value in self._data.items():
            if not cmath.isclose(value, other._data[key], rel_tol=self.rtol, abs_tol=self.atol):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        """Check exact equality of two ``SparseLabelOp`` instances

        Args:
            other: the second ``SparseLabelOp`` to compare with this instance.

        Returns:
            True if operators are equal, False if not.
        """
        if not isinstance(other, SparseLabelOp):
            return False
        return self.register_length == other._register_length and self._data == other._data

    def __getitem__(self, __k: str) -> complex:
        """Return a specified ``SparseLabelOp`` item"""
        return self._data.__getitem__(__k)

    def __len__(self) -> int:
        """Return length of ``SparseLabelOp``"""
        return self._data.__len__()

    def __iter__(self) -> Iterator[str]:
        """Iterate through ``SparseLabelOp`` items"""
        return self._data.__iter__()
