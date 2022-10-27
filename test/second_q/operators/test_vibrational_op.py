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

# import numpy as np
from ddt import ddt  # , data

from qiskit_nature.second_q.operators import VibrationalOp


@ddt
class TestVibrationalOp(QiskitNatureTestCase):
    """VibrationalOp tests."""

    op1 = VibrationalOp({"+_0_0 -_0_0": 1}, num_modes=1, num_modals=[1])

    op2 = VibrationalOp({"-_0_0 +_0_0": 2})
    op3 = VibrationalOp({"+_0_0 -_0_0": 1, "-_0_0 +_0_0": 2})

    def test_automatic_num_modes_and_num_modals(self):
        """Test operators with automatic num_modes and num_modals"""

        with self.subTest("Empty data"):
            op = VibrationalOp({"": 1})
            self.assertEqual(op.num_modes, 0)
            self.assertEqual(op.num_modals, [0])

        with self.subTest("Single mode and modal"):
            op = VibrationalOp({"+_0_0": 1})
            self.assertEqual(op.num_modes, 1)
            self.assertEqual(op.num_modals, [1])

        with self.subTest("Single mode and modal"):
            op = VibrationalOp({"+_0_0 +_1_0": 1})
            self.assertEqual(op.num_modes, 2)
            self.assertEqual(op.num_modals, [1, 1])

        with self.subTest("Single mode and modal"):
            op = VibrationalOp({"+_0_0 +_1_1": 1})
            self.assertEqual(op.num_modes, 2)
            self.assertEqual(op.num_modals, [1, 2])

        with self.subTest("Single mode and modal"):
            op = VibrationalOp({"+_0_0 +_1_1": 1}, num_modals=2)
            self.assertEqual(op.num_modes, 2)
            self.assertEqual(op.num_modals, [2, 2])

    #     with self.subTest("Mathematical operations"):
    #         self.assertEqual((op0 + op2).num_spin_orbitals, 2)
    #         self.assertEqual((op1 + op2).num_spin_orbitals, 2)
    #         self.assertEqual((op0 @ op2).num_spin_orbitals, 2)
    #         self.assertEqual((op1 @ op2).num_spin_orbitals, 2)
    #         self.assertEqual((op1 ^ op2).num_spin_orbitals, 3)

    #     with self.subTest("Equality"):
    #         op3 = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=3)
    #         self.assertEqual(op1, op3)
    #         self.assertTrue(op1.equiv(1.000001 * op3))

    def test_neg(self):
        """Test __neg__"""
        vib_op = -self.op1
        targ = VibrationalOp({"+_0_0 -_0_0": -1}, num_modes=1, num_modals=1)
        self.assertEqual(vib_op, targ)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        with self.subTest("rightmul"):
            vib_op = self.op1 * 2
            targ = VibrationalOp({"+_0_0 -_0_0": 2}, num_modes=1, num_modals=1)
            self.assertEqual(vib_op, targ)

        with self.subTest("left mul"):
            vib_op = (2 + 1j) * self.op3
            targ = VibrationalOp(
                {"+_0_0 -_0_0": (2 + 1j), "-_0_0 +_0_0": (4 + 2j)}, num_modes=1, num_modals=1
            )
            self.assertEqual(vib_op, targ)

    def test_div(self):
        """Test __truediv__"""
        vib_op = self.op1 / 2
        targ = VibrationalOp({"+_0_0 -_0_0": 0.5}, num_modes=1, num_modals=1)
        self.assertEqual(vib_op, targ)

    def test_add(self):
        """Test __add__"""
        vib_op = self.op1 + self.op2
        targ = self.op3
        self.assertEqual(vib_op, targ)

    def test_sub(self):
        """Test __sub__"""
        vib_op = self.op3 - self.op2
        targ = VibrationalOp({"+_0_0 -_0_0": 1, "-_0_0 +_0_0": 0}, num_modes=1, num_modals=[1])
        self.assertEqual(vib_op, targ)

    def test_compose(self):
        """Test operator composition"""
        with self.subTest("single compose"):
            vib_op = VibrationalOp(
                {"+_0_0 -_1_0": 1}, num_modes=2, num_modals=[1, 1]
            ) @ VibrationalOp({"-_0_0": 1}, num_modes=2, num_modals=1)
            targ = VibrationalOp({"+_0_0 -_1_0 -_0_0": 1}, num_modes=2, num_modals=[1, 1])
            self.assertEqual(vib_op, targ)

        with self.subTest("multi compose"):
            vib_op = VibrationalOp(
                {"+_0_0 +_1_0 -_1_0": 1, "-_0_0 +_0_0 -_1_0": 1}, num_modes=2, num_modals=[1, 1]
            ) @ VibrationalOp({"": 1, "-_0_0 +_1_0": 1}, num_modes=2, num_modals=[1, 1])
            vib_op = vib_op.simplify()
            targ = VibrationalOp(
                {
                    "+_0_0 +_1_0 -_1_0": 1,
                    "-_0_0 +_0_0 -_1_0": 1,
                    "+_0_0 +_1_0 -_0_0": 1,
                    "-_0_0 -_1_0 +_1_0": 1,
                },
                num_modes=2,
                num_modals=[1, 1],
            )
            self.assertEqual(vib_op, targ)

    def test_tensor(self):
        """Test tensor multiplication"""
        vib_op = self.op1.tensor(self.op2)
        targ = VibrationalOp({"+_0_0 -_0_0 -_1_0 +_1_0": 2}, num_modes=2, num_modals=1)
        self.assertEqual(vib_op, targ)

    def test_expand(self):
        """Test reversed tensor multiplication"""
        vib_op = self.op1.expand(self.op2)
        targ = VibrationalOp({"-_0_0 +_0_0 +_1_0 -_1_0": 2}, num_modes=2, num_modals=1)
        self.assertEqual(vib_op, targ)

    def test_pow(self):
        """Test __pow__"""
        with self.subTest("square trivial"):
            vib_op = (
                VibrationalOp({"+_0_0 +_1_0 -_1_0": 3, "-_0_0 +_0_0 -_1_0": 1}, num_modes=2) ** 2
            )
            vib_op = vib_op.simplify()
            targ = VibrationalOp.zero()
            self.assertEqual(vib_op, targ)

        with self.subTest("square nontrivial"):
            vib_op = (
                VibrationalOp(
                    {"+_0_0 +_1_0 -_1_0": 3, "+_0_0 -_0_0 -_1_0": 1}, num_modes=2, num_modals=[1, 1]
                )
                ** 2
            )
            vib_op = vib_op.simplify()
            targ = VibrationalOp({"+_0_0 -_1_0": 3}, num_modes=1, num_modals=[1])
            self.assertEqual(vib_op, targ)

        with self.subTest("3rd power"):
            vib_op = (3 * VibrationalOp.one()) ** 3
            targ = 27 * VibrationalOp.one()
            self.assertEqual(vib_op, targ)

        with self.subTest("0th power"):
            vib_op = (
                VibrationalOp(
                    {"+_0_0 +_1_0 -_1_0": 3, "-_0_0 +_0_0 -_1_0": 1}, num_modes=2, num_modals=[1, 1]
                )
                ** 0
            )
            vib_op = vib_op.simplify()
            targ = VibrationalOp.one()
            self.assertEqual(vib_op, targ)

    def test_adjoint(self):
        """Test adjoint method"""
        vib_op = VibrationalOp(
            {"": 1j, "+_0_0 +_1_0 -_1_0": 3, "+_0_0 -_0_0 -_1_0": 1, "-_0_0 -_1_0": 2 + 4j},
            num_modes=3,
        ).adjoint()
        targ = VibrationalOp(
            {"": -1j, "+_1_0 -_1_0 -_0_0": 3, "+_1_0 +_0_0 -_0_0": 1, "+_1_0 +_0_0": 2 - 4j},
            num_modes=3,
        )
        self.assertEqual(vib_op, targ)

    def test_simplify(self):
        """Test simplify"""
        with self.subTest("simplify integer"):
            vib_op = VibrationalOp(
                {"+_0_0 -_0_0": 1, "+_0_0 -_0_0 +_0_0 -_0_0": 1}, num_modes=1, num_modals=[1]
            )
            simplified_op = vib_op.simplify()
            targ = VibrationalOp({"+_0_0 -_0_0": 2}, num_modes=1, num_modals=[1])
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify complex"):
            vib_op = VibrationalOp(
                {"+_0_0 -_0_0": 1, "+_0_0 -_0_0 +_0_0 -_0_0": 1j}, num_modes=1, num_modals=[1]
            )
            simplified_op = vib_op.simplify()
            targ = VibrationalOp({"+_0_0 -_0_0": 1 + 1j}, num_modes=1, num_modals=[1])
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify doesn't reorder"):
            vib_op = VibrationalOp({"-_0_0 +_1_0": 1 + 0j}, num_modes=2, num_modals=[1, 1])
            simplified_op = vib_op.simplify()
            self.assertEqual(simplified_op, vib_op)

            vib_op = VibrationalOp({"-_1_0 +_0_0": 1 + 0j}, num_modes=2, num_modals=[1, 1])
            simplified_op = vib_op.simplify()
            self.assertEqual(simplified_op, vib_op)

        with self.subTest("simplify zero"):
            vib_op = self.op1 - self.op1
            simplified_op = vib_op.simplify()
            targ = VibrationalOp.zero()
            self.assertEqual(simplified_op, targ)

    #     def test_equiv(self):
    #         """test equiv"""
    #         prev_atol = FermionicOp.atol
    #         prev_rtol = FermionicOp.rtol
    #         op3 = self.op1 + (1 + 0.00005) * self.op2
    #         self.assertFalse(op3.equiv(self.op3))
    #         FermionicOp.atol = 1e-4
    #         FermionicOp.rtol = 1e-4
    #         self.assertTrue(op3.equiv(self.op3))
    #         FermionicOp.atol = prev_atol
    #         FermionicOp.rtol = prev_rtol

    def test_induced_norm(self):
        """Test induced norm."""
        op = 3 * VibrationalOp({"+_0_0": 1}, num_modes=1) + 4j * VibrationalOp(
            {"-_0_0": 1}, num_modes=1
        )
        self.assertAlmostEqual(op.induced_norm(), 7.0)
        self.assertAlmostEqual(op.induced_norm(2), 5.0)


if __name__ == "__main__":
    unittest.main()
