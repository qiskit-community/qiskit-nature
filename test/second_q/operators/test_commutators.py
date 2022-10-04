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

"""Test for commutators"""

from __future__ import annotations

from typing import Collection, Iterator

import unittest
from test import QiskitNatureTestCase

from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.operators.commutators import (
    commutator,
    anti_commutator,
    double_commutator
)

op1 = {
    "+_0 -_1": 0.0,
    "+_0 -_2": 1.0,
    }

op2 = {
    "+_0 -_1": 0.5,
    "+_0 -_2": 1.0,
}

op3 = {
    "+_0 -_1": 0.5,
    "+_0 -_3": 3.0,
}

opComplex = {
    "+_0 -_1": 0.5 + 1j,
    "+_0 -_2": 1.0,
}

class DummySparseLabelOp(SparseLabelOp):
    """Dummy SparseLabelOp for testing purposes"""

    @classmethod
    def _validate_keys(cls, keys: Collection[str], register_length: int) -> None:
        pass

    def terms(self) -> Iterator[tuple[list[tuple[str, int]], complex]]:
        pass

    def transpose(self) -> SparseLabelOp:
        return self

    def compose(self, other, qargs=None, front=False) -> SparseLabelOp:
        return self

    def tensor(self, other) -> SparseLabelOp:
        return self

    def expand(self, other) -> SparseLabelOp:
        return self

    # pylint: disable=unused-argument
    def simplify(self, *, atol: float | None = None) -> SparseLabelOp:
        return self


class TestCommutators(QiskitNatureTestCase):
    """Commutators tests."""

    def test_commutator(self):
        """Test commutator method"""
        op_a = DummySparseLabelOp(op1, 2)
        op_b = DummySparseLabelOp(op2, 2)

        print(commutator(op_a, op_b))

    def test_anti_commutator(self):
        """Test anti commutator method"""
        op_a = DummySparseLabelOp(op1, 2)
        op_b = DummySparseLabelOp(op2, 2)

        print(anti_commutator(op_a, op_b))

    def test_double_commutator(self):
        """Test double commutator method"""
        op_a = DummySparseLabelOp(op1, 2)
        op_b = DummySparseLabelOp(op2, 2)
        op_c = DummySparseLabelOp(op3, 2)

        print(double_commutator(op_a, op_b, op_c))


if __name__ == "__main__":
    unittest.main()
