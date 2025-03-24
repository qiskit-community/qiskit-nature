# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for MixedOp"""

import unittest
from test import QiskitNatureTestCase

from qiskit_nature.second_q.operators import FermionicOp, MixedOp


class TestFermionicOp(QiskitNatureTestCase):
    """FermionicOp tests."""

    op1 = FermionicOp({"+_0 -_0": 1})
    op2 = FermionicOp({"-_0 +_0": 2})
    mop1_h1 = MixedOp({("h1",): [(2.0, op1)]})
    mop2_h1 = MixedOp({("h1",): [(3.0, op2)]})
    mop2_h2 = MixedOp({("h2",): [(3.0, op2)]})
    sumop2 = MixedOp({("h1",): [(1, FermionicOp({"+_0 -_0": 2, "-_0 +_0": 3}))]})

    def test_neg(self):
        """Test __neg__"""
        minus_mop1 = -self.mop1_h1
        target_v1 = MixedOp({("h1",): [(-2.0, self.op1)]})
        self.assertEqual(minus_mop1, target_v1)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        with self.subTest("rightmul"):
            minus_mop1 = self.mop1_h1 * 2.0
            target_v1 = MixedOp({("h1",): [(4.0, self.op1)]})
            self.assertEqual(minus_mop1, target_v1)

        with self.subTest("leftmul"):
            minus_mop1 = (2.0 + 1.0j) * self.mop1_h1
            target_v1 = MixedOp({("h1",): [((4.0 + 2.0j), self.op1)]})
            self.assertEqual(minus_mop1, target_v1)

    def test_div(self):
        """Test __truediv__"""
        fer_op = self.op1 / 2
        target = FermionicOp({"+_0 -_0": 0.5}, num_spin_orbitals=1)
        self.assertEqual(fer_op, target)

    def test_add(self):
        """Test __add__"""
        with self.subTest("same hilbert space"):
            sum_mop = self.mop1_h1 + self.mop2_h1
            target = MixedOp({("h1",): [(2, self.op1), (3, self.op2)]})
            self.assertEqual(sum_mop, target)

        with self.subTest("different hilbert space"):
            sum_mop = self.mop1_h1 + self.mop2_h2
            target = MixedOp({("h1",): [(2.0, self.op1)], ("h2",): [(3.0, self.op2)]})
            self.assertEqual(sum_mop, target)

    def test_sub(self):
        """Test __sub__"""
        with self.subTest("same hilbert space"):
            sum_mop = self.mop1_h1 - self.mop2_h1
            target = MixedOp({("h1",): [(2, self.op1), (-3, self.op2)]})
            self.assertEqual(sum_mop, target)

        with self.subTest("different hilbert space"):
            sum_mop = self.mop1_h1 - self.mop2_h2
            target = MixedOp({("h1",): [(2.0, self.op1)], ("h2",): [(-3.0, self.op2)]})
            self.assertEqual(sum_mop, target)

    def test_compose(self):
        """Test operator composition"""
        with self.subTest("same hilbert spaces"):
            composed_op = MixedOp.compose(self.mop1_h1, self.mop2_h1)
            target = MixedOp({("h1", "h1"): [(6.0, self.op1, self.op2)]})
            self.assertEqual(composed_op, target)

        with self.subTest("different hilbert spaces"):
            composed_op = MixedOp.compose(self.mop1_h1, self.mop2_h2)
            target = MixedOp({("h1", "h2"): [(6.0, self.op1, self.op2)]})
            self.assertEqual(composed_op, target)


if __name__ == "__main__":
    unittest.main()
