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
from typing import Dict, Iterator
import cmath
import numpy as np

from qiskit.quantum_info.operators.mixins import LinearMixin, AdjointMixin, TolerancesMixin


class SparseLabelOp(LinearMixin, AdjointMixin, TolerancesMixin):
    """The Sparse Label Operator base class."""

    def __init__(self, data: Dict[str, complex], register_length: int | None = None):
        self._data = data
        self._register_length = register_length

    def _add(self, other: SparseLabelOp, qargs=None) -> SparseLabelOp:
        """Return Operator addition of self and other.

        Args:
            other: the second ``SparseLabelOp`` to add to the first.

        Returns:
            SparseLabelOp: the new summed ``SparseLabelOp``.

        Raises:
            ValueError: when ``qargs`` argument is not ``None``
        """
        if qargs is not None:
            raise ValueError("`qargs` argument should be `None`")

        new_data = self._data.copy()

        for key, value in other._data.items():
            if key in new_data.keys():
                new_data[key] += value
            else:
                new_data[key] = value

        return SparseLabelOp(new_data)

    def _multiply(self, other: complex) -> SparseLabelOp:
        """Return scalar multiplication of self and other.

        Args:
            other: the number to multiply the ``SparseLabelOp`` values by.

        Returns:
            SparseLabelOp: the newly multiplied ``SparseLabelOp``.

        Raises:
            TypeError: if ``other`` is not compatible type (int, float or complex)
        """
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SparseLabelOp' and '{type(other).__name__}'"
            )
        new_data = {}

        for key, value in self._data.items():
            new_data[key] = value * other

        return SparseLabelOp(new_data)

    def conjugate(self) -> SparseLabelOp:
        """Return the conjugate of the ``SparseLabelOp``

        Returns:
            SparseLabelOp: the complex conjugate of the starting ``SparseLabelOp``.
        """
        new_data = {}

        for key, value in self._data.items():
            new_data[key] = np.conjugate(value)

        return SparseLabelOp(new_data)

    def transpose(self) -> SparseLabelOp:
        """This method has no effect on ``SparseLabelOp`` and returns itself.

        Returns:
            SparseLabelOp: the initial ``SparseLabelOp``.
        """
        return self

    def equiv(self, other: SparseLabelOp) -> bool:
        """Check equivalence of two ``SparseLabelOp`` instances to an accepted tolerance

        Args:
            other: the second ``SparseLabelOp`` to compare the first with.

        Returns:
            bool: True if operators are equal to, False if not.
        """
        if set(self._data.keys()) != set(other._data.keys()):
            return False
        for key, val in self._data.items():
            if not cmath.isclose(val, other._data[key], rel_tol=self.rtol, abs_tol=self.atol):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        """Check exact equality of two ``SparseLabelOp`` instances

        Args:
            other: the second ``SparseLabelOp`` to compare the first with.

        Returns:
            bool: True if operators are equal, False if not.
        """
        if not isinstance(other, SparseLabelOp):
            return NotImplemented
        if set(self._data.keys()) != set(other._data.keys()):
            return False
        for key, val in self._data.items():
            if not val == other._data[key]:
                return False
        return True

    def __iter__(self) -> Iterator[tuple[str, complex]]:
        """Iterate through ``SparseLabelOp`` items"""
        for key, value in self._data.items():
            yield key, value
