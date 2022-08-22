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
from typing import Mapping, Iterator
from numbers import Number
import cmath
import numpy as np

from qiskit.quantum_info.operators.mixins import LinearMixin, AdjointMixin, TolerancesMixin


class SparseLabelOp(LinearMixin, AdjointMixin, TolerancesMixin):
    """The Sparse Label Operator base class."""

    def __init__(self, data: Mapping[str, complex], register_length: int, **kwargs):
        """
        Args:
            data: Data for operator comprising string key to coeff value mapping
            register_length: Length of register needed for data
        """
        self._data: Mapping[str, complex] = {}
        if kwargs.get("_dont_copy", False):
            self._data = data  # Store by reference
        else:
            self._data = dict(data.items())  # Store by value
        self._register_length = register_length

    def _add(self, other: SparseLabelOp, qargs=None) -> SparseLabelOp:
        """Return Operator addition of self and other.

        Args:
            other: the second ``SparseLabelOp`` to add to the first.

        Returns:
            the new summed ``SparseLabelOp``.

        Raises:
            ValueError: when ``qargs`` argument is not ``None``
        """
        if qargs is not None:
            raise ValueError(f"The `qargs` argument must be `None`, not {qargs}")

        if not isinstance(other, SparseLabelOp):
            raise ValueError(
                f"Unsupported operand type(s) for +: 'SparseLabelOp' and '{type(other).__name__}'"
            )
        
        new_data = {key: value + other._data.get(key, 0) for key, value in self._data.items()}
        other_unique = {key: other._data[key] for key in other._data.keys() - self._data.keys()}
        new_data.update(other_unique)

        register_length = max(self._register_length, other._register_length)

        return SparseLabelOp(new_data, register_length, _dont_copy=True)

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
        new_data = {
            key: val * other for key, val in self._data.items()
        }

        return SparseLabelOp(new_data, self._register_length, _dont_copy=True)

    def conjugate(self) -> SparseLabelOp:
        """Returns the conjugate of the ``SparseLabelOp``

        Returns:
            the complex conjugate of the starting ``SparseLabelOp``.
        """
        new_data = {key: np.conjugate(val) for key, val in self._data.items()}

        return SparseLabelOp(new_data, self._register_length, _dont_copy=True)

    def transpose(self) -> SparseLabelOp:
        """This method has no effect on ``SparseLabelOp`` and returns itself.

        Returns:
            the initial ``SparseLabelOp``.
        """
        return self

    def equiv(self, other: SparseLabelOp) -> bool:
        """Check equivalence of two ``SparseLabelOp`` instances to an accepted tolerance

        Args:
            other: the second ``SparseLabelOp`` to compare with this instance.

        Returns:
            True if operators are equal to, False if not.
        """
        if not isinstance(other, SparseLabelOp):
            return False
        if self._register_length != other._register_length:
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
        return self._register_length == other._register_length and self._data == other._data

    def __iter__(self) -> Iterator[tuple[str, complex]]:
        """Iterate through ``SparseLabelOp`` items"""
        for key, value in self._data.items():
            yield key, value
