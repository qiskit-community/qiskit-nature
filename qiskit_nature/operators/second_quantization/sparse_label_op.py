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

"""The Sparse Label Operator base interface."""

from __future__ import annotations
from typing import Dict, Iterator
import math
from xmlrpc.client import Boolean
import numpy as np

from qiskit.quantum_info.operators.mixins import LinearMixin, AdjointMixin, TolerancesMixin


class SparseLabelOp(LinearMixin, AdjointMixin, TolerancesMixin):
    """The Sparse Label Operator base interface."""

    def __init__(self, data: Dict[str, complex], register_length: int = None):
        self._data = data
        self._register_length = register_length

    def _add(self, other, qargs=None) -> SparseLabelOp:
        """Return Operator addition of self and other"""
        new_data = self._data.copy()

        for key, value in other._data.items():
            if key in new_data.keys():
                new_data[key] += value
            else:
                new_data[key] = value

        return SparseLabelOp(new_data)

    def _multiply(self, other: complex) -> SparseLabelOp:
        """Return scalar multiplication of self and other"""
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SparseLabelOp' and '{type(other).__name__}'"
            )
        new_data = self._data.copy()

        for key, value in self._data.items():
            new_data[key] = value * other

        return SparseLabelOp(new_data)

    def conjugate(self) -> SparseLabelOp:
        """Return the conjugate of the ``SparseLabelOp``"""
        new_data = self._data.copy()

        for key, value in self._data.items():
            new_data[key] = np.conjugate(value)

        return SparseLabelOp(new_data)

    def transpose(self) -> SparseLabelOp:
        return self

    def equiv(self, other: SparseLabelOp) -> Boolean:
        """Check equivalence of two ``SparseLabelOp`` instances to an accepted tolerance

        Args:
            other: the second ``SparseLabelOp`` to compare the first with.

        Returns:
            Bool: True if operators are equal to, False if not.
        """
        if set(self._data.keys()) != set(other._data.keys()):
            return False
        for key, val in self._data.items():
            if not math.isclose(val, other._data[key], rel_tol=self.rtol, abs_tol=self.atol):
                return False
        return True

    def __eq__(self, other: SparseLabelOp) -> Boolean:
        """Check exact equality of two ``SparseLabelOp`` instances

        Args:
            other: the second ``SparseLabelOp`` to compare the first with.

        Returns:
            Bool: True if operators are equal, False if not.
        """
        if set(self._data.keys()) != set(other._data.keys()):
            return False
        for key, val in self._data.items():
            if not val == other._data[key]:
                return False
        return True

    def __iter__(self) -> Iterator[SparseLabelOp]:
        """Iterate through ``SparseLabelOp`` items"""
        return iter(self._data.items())
