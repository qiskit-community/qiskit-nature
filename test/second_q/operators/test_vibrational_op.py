# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test for VibrationalOp"""

import unittest
from test import QiskitNatureTestCase

from qiskit_nature.second_q.operators import VibrationalOp


class TestVibrationalOp(QiskitNatureTestCase):
    """VibrationalOp tests."""

    op1 = VibrationalOp({"+_0_0 -_0_0": 1}, num_modals=[1])

    op2 = VibrationalOp({"-_0_0 +_0_0": 2})
    op3 = VibrationalOp({"+_0_0 -_0_0": 1, "-_0_0 +_0_0": 2})

    def test_automatic_num_modals(self):
        """Test operators with automatic num_modals"""

        with self.subTest("Empty data"):
            op = VibrationalOp({"": 1})
            self.assertEqual(op.num_modals, [])

        with self.subTest("Single mode and modal"):
            op = VibrationalOp({"+_0_0": 1})
            self.assertEqual(op.num_modals, [1])

        with self.subTest("Single mode and modal"):
            op = VibrationalOp({"+_0_0 +_1_0": 1})
            self.assertEqual(op.num_modals, [1, 1])

        with self.subTest("Single mode and modal"):
            op = VibrationalOp({"+_0_0 +_1_1": 1})
            self.assertEqual(op.num_modals, [1, 2])

        with self.subTest("Single mode and modal"):
            op = VibrationalOp({"+_0_0 +_1_1": 1}, num_modals=[0, 0])
            self.assertEqual(op.num_modals, [1, 2])

    def test_neg(self):
        """Test __neg__"""
        vib_op = -self.op1
        targ = VibrationalOp({"+_0_0 -_0_0": -1}, num_modals=[1])
        self.assertEqual(vib_op, targ)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        with self.subTest("rightmul"):
            vib_op = self.op1 * 2
            targ = VibrationalOp({"+_0_0 -_0_0": 2}, num_modals=[1])
            self.assertEqual(vib_op, targ)

        with self.subTest("left mul"):
            vib_op = (2 + 1j) * self.op3
            targ = VibrationalOp({"+_0_0 -_0_0": (2 + 1j), "-_0_0 +_0_0": (4 + 2j)}, num_modals=[1])
            self.assertEqual(vib_op, targ)

    def test_div(self):
        """Test __truediv__"""
        vib_op = self.op1 / 2
        targ = VibrationalOp({"+_0_0 -_0_0": 0.5}, num_modals=[1])
        self.assertEqual(vib_op, targ)

    def test_add(self):
        """Test __add__"""
        vib_op = self.op1 + self.op2
        targ = self.op3
        self.assertEqual(vib_op, targ)

    def test_sub(self):
        """Test __sub__"""
        vib_op = self.op3 - self.op2
        targ = VibrationalOp({"+_0_0 -_0_0": 1, "-_0_0 +_0_0": 0}, num_modals=[1])
        self.assertEqual(vib_op, targ)

    def test_compose(self):
        """Test operator composition"""
        with self.subTest("single compose"):
            vib_op = VibrationalOp({"+_0_0 -_1_0": 1}, num_modals=[1, 1]) @ VibrationalOp(
                {"-_0_0": 1}, num_modals=[1, 1]
            )
            targ = VibrationalOp({"+_0_0 -_1_0 -_0_0": 1}, num_modals=[1, 1])
            self.assertEqual(vib_op, targ)

        with self.subTest("multi compose"):
            vib_op = VibrationalOp(
                {"+_0_0 +_1_0 -_1_0": 1, "-_0_0 +_0_0 -_1_0": 1}, num_modals=[1, 1]
            ) @ VibrationalOp({"": 1, "-_0_0 +_1_0": 1}, num_modals=[1, 1])
            vib_op = vib_op.simplify()
            targ = VibrationalOp(
                {
                    "+_0_0 +_1_0 -_1_0": 1,
                    "-_0_0 +_0_0 -_1_0": 1,
                    "+_0_0 +_1_0 -_0_0": 1,
                    "-_0_0 -_1_0 +_1_0": 1,
                },
                num_modals=[1, 1],
            )
            self.assertEqual(vib_op, targ)

    def test_tensor(self):
        """Test tensor multiplication"""
        vib_op = self.op1.tensor(self.op2)
        targ = VibrationalOp({"+_0_0 -_0_0 -_1_0 +_1_0": 2}, num_modals=[1, 1])
        self.assertEqual(vib_op, targ)

    def test_expand(self):
        """Test reversed tensor multiplication"""
        vib_op = self.op1.expand(self.op2)
        targ = VibrationalOp({"-_0_0 +_0_0 +_1_0 -_1_0": 2}, num_modals=[1, 1])
        self.assertEqual(vib_op, targ)

    def test_pow(self):
        """Test __pow__"""
        with self.subTest("square trivial"):
            vib_op = (
                VibrationalOp({"+_0_0 +_1_0 -_1_0": 3, "-_0_0 +_0_0 -_1_0": 1}, num_modals=[1, 1])
                ** 2
            )
            vib_op = vib_op.simplify()
            targ = VibrationalOp.zero()
            self.assertEqual(vib_op, targ)

        with self.subTest("square nontrivial"):
            vib_op = (
                VibrationalOp({"+_0_0 +_1_0 -_1_0": 3, "+_0_0 -_0_0 -_1_0": 1}, num_modals=[1, 1])
                ** 2
            )
            vib_op = vib_op.simplify()
            targ = VibrationalOp({"+_0_0 -_1_0": 3}, num_modals=[1])
            self.assertEqual(vib_op, targ)

        with self.subTest("3rd power"):
            vib_op = (3 * VibrationalOp.one()) ** 3
            targ = 27 * VibrationalOp.one()
            self.assertEqual(vib_op, targ)

        with self.subTest("0th power"):
            vib_op = (
                VibrationalOp({"+_0_0 +_1_0 -_1_0": 3, "-_0_0 +_0_0 -_1_0": 1}, num_modals=[1, 1])
                ** 0
            )
            vib_op = vib_op.simplify()
            targ = VibrationalOp.one()
            self.assertEqual(vib_op, targ)

    def test_adjoint(self):
        """Test adjoint method"""
        vib_op = VibrationalOp(
            {"": 1j, "+_0_0 +_1_0 -_1_0": 3, "+_0_0 -_0_0 -_1_0": 1, "-_0_0 -_1_0": 2 + 4j},
            num_modals=[1, 1, 1],
        ).adjoint()
        targ = VibrationalOp(
            {"": -1j, "+_1_0 -_1_0 -_0_0": 3, "+_1_0 +_0_0 -_0_0": 1, "+_1_0 +_0_0": 2 - 4j},
            num_modals=[1, 1, 1],
        )
        self.assertEqual(vib_op, targ)

    def test_simplify(self):
        """Test simplify"""
        with self.subTest("simplify integer"):
            vib_op = VibrationalOp({"+_0_0 -_0_0": 1, "+_0_0 -_0_0 +_0_0 -_0_0": 1}, num_modals=[1])
            simplified_op = vib_op.simplify()
            targ = VibrationalOp({"+_0_0 -_0_0": 2}, num_modals=[1])
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify complex"):
            vib_op = VibrationalOp(
                {"+_0_0 -_0_0": 1, "+_0_0 -_0_0 +_0_0 -_0_0": 1j}, num_modals=[1]
            )
            simplified_op = vib_op.simplify()
            targ = VibrationalOp({"+_0_0 -_0_0": 1 + 1j}, num_modals=[1])
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify doesn't reorder"):
            vib_op = VibrationalOp({"-_0_0 +_1_0": 1 + 0j}, num_modals=[1, 1])
            simplified_op = vib_op.simplify()
            self.assertEqual(simplified_op, vib_op)

            vib_op = VibrationalOp({"-_1_0 +_0_0": 1 + 0j}, num_modals=[1, 1])
            simplified_op = vib_op.simplify()
            self.assertEqual(simplified_op, vib_op)

        with self.subTest("simplify zero"):
            vib_op = self.op1 - self.op1
            simplified_op = vib_op.simplify()
            targ = VibrationalOp.zero()
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify commutes with normal_order"):
            self.assertEqual(self.op2.simplify().normal_order(), self.op2.normal_order().simplify())

        with self.subTest("simplify + index order"):
            orig = VibrationalOp({"+_1_0 -_0_0 +_0_0 -_0_0": 1, "-_0_0 +_1_0": 2})
            vib_op = orig.simplify().index_order()
            targ = VibrationalOp({"-_0_0 +_1_0": 3})
            self.assertEqual(vib_op, targ)

    def test_equiv(self):
        """test equiv"""
        prev_atol = VibrationalOp.atol
        prev_rtol = VibrationalOp.rtol
        op3 = self.op1 + (1 + 0.00005) * self.op2
        self.assertFalse(op3.equiv(self.op3))
        VibrationalOp.atol = 1e-4
        VibrationalOp.rtol = 1e-4
        self.assertTrue(op3.equiv(self.op3))
        VibrationalOp.atol = prev_atol
        VibrationalOp.rtol = prev_rtol

    def test_induced_norm(self):
        """Test induced norm."""
        op = 3 * VibrationalOp({"+_0_0": 1}, num_modals=[0]) + 4j * VibrationalOp(
            {"-_0_0": 1}, num_modals=[0]
        )
        self.assertAlmostEqual(op.induced_norm(), 7.0)
        self.assertAlmostEqual(op.induced_norm(2), 5.0)

    def test_normal_order(self):
        """test normal_order method"""
        with self.subTest("Test for creation operator"):
            orig = VibrationalOp({"+_0_0": 1})
            vib_op = orig.normal_order()
            self.assertEqual(vib_op, orig)

        with self.subTest("Test for annihilation operator"):
            orig = VibrationalOp({"-_0_0": 1})
            vib_op = orig.normal_order()
            self.assertEqual(vib_op, orig)

        with self.subTest("Test for number operator"):
            orig = VibrationalOp({"+_0_0 -_0_0": 1})
            vib_op = orig.normal_order()
            self.assertEqual(vib_op, orig)

        with self.subTest("Test for empty operator"):
            orig = VibrationalOp({"-_0_0 +_0_0": 1})
            vib_op = orig.normal_order()
            targ = VibrationalOp({"+_0_0 -_0_0": 1})
            self.assertEqual(vib_op, targ)

        with self.subTest("Test for multiple operators 1"):
            orig = VibrationalOp({"-_0_0 +_1_0": 1})
            vib_op = orig.normal_order()
            targ = VibrationalOp({"+_1_0 -_0_0": 1})
            self.assertEqual(vib_op, targ)

        with self.subTest("Test for multiple operators 2"):
            orig = VibrationalOp({"-_0_0 +_0_0 +_1_0 -_2_0": 1})
            vib_op = orig.normal_order()
            targ = VibrationalOp({"+_0_0 +_1_0 -_0_0 -_2_0": 1})
            self.assertEqual(vib_op, targ)

        with self.subTest("Test normal ordering simplifies"):
            orig = VibrationalOp({"-_0_0 +_1_0": 1, "+_1_0 -_0_0": 1, "+_0_0": 0.0})
            vib_op = orig.normal_order()
            targ = VibrationalOp({"+_1_0 -_0_0": 2})
            self.assertEqual(vib_op, targ)

        with self.subTest("Test with multiple modals 1"):
            orig = VibrationalOp({"-_0_0 +_0_1": 1})
            vib_op = orig.normal_order()
            targ = VibrationalOp({"+_0_1 -_0_0": 1})
            self.assertEqual(vib_op, targ)

        with self.subTest("Test with multiple modals 2"):
            orig = VibrationalOp({"+_0_1 -_1_0": 1})
            vib_op = orig.normal_order()
            self.assertEqual(vib_op, orig)

        with self.subTest("Test with multiple modals 3"):
            orig = VibrationalOp({"-_1_1 +_1_0 +_0_1 -_0_0": 1})
            vib_op = orig.normal_order()
            targ = VibrationalOp({"+_0_1 +_1_0 -_0_0 -_1_1": 1})
            self.assertEqual(vib_op, targ)

    def test_index_order(self):
        """test index_order method"""
        with self.subTest("Test for creation operator"):
            orig = VibrationalOp({"+_0_0": 1})
            vib_op = orig.index_order()
            self.assertEqual(vib_op, orig)

        with self.subTest("Test for annihilation operator"):
            orig = VibrationalOp({"-_0_0": 1})
            vib_op = orig.index_order()
            self.assertEqual(vib_op, orig)

        with self.subTest("Test for number operator"):
            orig = VibrationalOp({"+_0_0 -_0_0": 1})
            vib_op = orig.index_order()
            self.assertEqual(vib_op, orig)

        with self.subTest("Test for empty operator"):
            orig = VibrationalOp({"-_0_0 +_0_0": 1})
            vib_op = orig.index_order()
            self.assertEqual(vib_op, orig)

        with self.subTest("Test for multiple operators 1"):
            orig = VibrationalOp({"+_1_0 -_0_0": 1})
            vib_op = orig.index_order()
            targ = VibrationalOp({"-_0_0 +_1_0": 1})
            self.assertEqual(vib_op, targ)

        with self.subTest("Test for multiple operators 2"):
            orig = VibrationalOp({"+_2_0 -_0_0 +_1_0 -_0_0": 1, "-_0_0 +_1_0": 2})
            vib_op = orig.index_order()
            targ = VibrationalOp({"-_0_0 -_0_0 +_1_0 +_2_0": 1, "-_0_0 +_1_0": 2})
            self.assertEqual(vib_op, targ)

        with self.subTest("Test index ordering simplifies"):
            orig = VibrationalOp({"-_0_0 +_1_0": 1, "+_1_0 -_0_0": 1, "+_0_0": 0.0})
            vib_op = orig.index_order()
            targ = VibrationalOp({"-_0_0 +_1_0": 2})
            self.assertEqual(vib_op, targ)

        with self.subTest("index order + simplify"):
            orig = VibrationalOp({"+_1_0 -_0_0 +_0_0 -_0_0": 1, "-_0_0 +_1_0": 2})
            vib_op = orig.index_order().simplify()
            targ = VibrationalOp({"-_0_0 +_1_0": 3})
            self.assertEqual(vib_op, targ)

        with self.subTest("Test with multiple modals 1"):
            orig = VibrationalOp({"+_0_1 -_0_0": 1})
            vib_op = orig.index_order()
            targ = VibrationalOp({"-_0_0 +_0_1": 1})
            self.assertEqual(vib_op, targ)

        with self.subTest("Test with multiple modals 2"):
            orig = VibrationalOp({"+_0_1 -_1_0": 1})
            vib_op = orig.index_order()
            self.assertEqual(vib_op, orig)

        with self.subTest("Test with multiple modals 3"):
            orig = VibrationalOp({"-_1_1 +_1_0 +_0_1 -_0_0": 1})
            vib_op = orig.index_order()
            targ = VibrationalOp({"-_0_0 +_0_1 +_1_0 -_1_1": 1})
            self.assertEqual(vib_op, targ)


if __name__ == "__main__":
    unittest.main()
