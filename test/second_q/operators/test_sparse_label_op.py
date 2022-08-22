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

"""Test for SparseLabelOp"""

import unittest
from test import QiskitNatureTestCase

from qiskit_nature.second_q.operators import SparseLabelOp

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
    def transpose(self) -> SparseLabelOp:
        return self


class TestDummySparseLabelOp(QiskitNatureTestCase):
    """DummySparseLabelOp tests."""

    def test_add(self):
        """Test add method"""
        with self.subTest("real + real"):
            test_op = DummySparseLabelOp(op1, 2) + DummySparseLabelOp(op2, 2)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.5,
                    "+_0 -_2": 2.0,
                },
                2,
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("complex + real"):
            test_op = DummySparseLabelOp(op2, 2) + DummySparseLabelOp(opComplex, 2)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 1.0 + 1j,
                    "+_0 -_2": 2.0,
                },
                2,
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("complex + complex"):
            test_op = DummySparseLabelOp(opComplex, 2) + DummySparseLabelOp(opComplex, 2)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 1.0 + 2j,
                    "+_0 -_2": 2.0,
                },
                2,
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("new key"):
            test_op = DummySparseLabelOp(op1, 2) + DummySparseLabelOp(op3, 2)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.5,
                    "+_0 -_2": 1.0,
                    "+_0 -_3": 3.0,
                },
                2,
            )

            self.assertEqual(test_op, target_op)

    def test_mul(self):
        """Test scalar multiplication method"""
        with self.subTest("real * real"):
            test_op = DummySparseLabelOp(op1, 2) * 2
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.0,
                    "+_0 -_2": 2.0,
                },
                2,
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("complex * real"):
            test_op = DummySparseLabelOp(opComplex, 2) * 2
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 1.0 + 2j,
                    "+_0 -_2": 2.0,
                },
                2,
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("real * complex"):
            test_op = DummySparseLabelOp(op2, 2) * (0.5 + 1j)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.25 + 0.5j,
                    "+_0 -_2": 0.5 + 1j,
                },
                2,
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("complex * complex"):
            test_op = DummySparseLabelOp(opComplex, 2) * (0.5 + 1j)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": -0.75 + 1j,
                    "+_0 -_2": 0.5 + 1j,
                },
                2,
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("raises TypeError"):
            with self.assertRaises(TypeError):
                _ = DummySparseLabelOp(op1, 2) * "something"

    def test_adjoint(self):
        """Test adjoint method"""
        test_op = DummySparseLabelOp(opComplex, 2).adjoint()
        target_op = DummySparseLabelOp(
            {
                "+_0 -_1": 0.5 - 1j,
                "+_0 -_2": 1.0,
            },
            2,
        )
        self.assertEqual(test_op, target_op)

    def test_conjugate(self):
        """Test conjugate method"""
        test_op = DummySparseLabelOp(opComplex, 2).conjugate()
        target_op = DummySparseLabelOp(
            {
                "+_0 -_1": 0.5 - 1j,
                "+_0 -_2": 1.0,
            },
            2,
        )
        self.assertEqual(test_op, target_op)

    def test_eq(self):
        """test __eq__ method"""
        with self.subTest("equal"):
            test_op = DummySparseLabelOp(op1, 2) == DummySparseLabelOp(op1, 2)
            self.assertTrue(test_op)

        with self.subTest("not equal - keys"):
            test_op = DummySparseLabelOp(op1, 2) == DummySparseLabelOp(
                {
                    "+_0 -_1": 0.0,
                    "+_0 -_3": 1.0,
                },
                2,
            )
            self.assertFalse(test_op)

        with self.subTest("not equal - values"):
            test_op = DummySparseLabelOp(op1, 2) == DummySparseLabelOp(op2, 2)
            self.assertFalse(test_op)

        with self.subTest("not equal - tolerance"):
            test_op = DummySparseLabelOp(op1, 2) == DummySparseLabelOp(
                {
                    "+_0 -_1": 0.000000001,
                    "+_0 -_2": 1.0,
                },
                2,
            )

            self.assertFalse(test_op)

    def test_equiv(self):
        """test equiv method"""
        with self.subTest("not equivalent - tolerances"):
            test_op = DummySparseLabelOp(op1, 2).equiv(
                DummySparseLabelOp(
                    {
                        "+_0 -_1": 0.000001,
                        "+_0 -_2": 1.0,
                    },
                    2,
                )
            )

            self.assertFalse(test_op)

        with self.subTest("not equivalent - keys"):
            test_op = DummySparseLabelOp(op1, 2).equiv(
                DummySparseLabelOp(
                    {
                        "+_0 -_1": 0.0,
                        "+_0 -_3": 1.0,
                    },
                    2,
                )
            )

            self.assertFalse(test_op)

        with self.subTest("equivalent"):
            test_op = DummySparseLabelOp(op1, 2).equiv(
                DummySparseLabelOp(
                    {
                        "+_0 -_1": 0.000000001,
                        "+_0 -_2": 1.0,
                    },
                    2,
                )
            )

            self.assertTrue(test_op)

    def test_iter(self):
        """test __iter__ method"""
        test_op = iter(DummySparseLabelOp(op1, 2))

        self.assertEqual(next(test_op), ("+_0 -_1", 0.0))
        self.assertEqual(next(test_op), ("+_0 -_2", 1.0))


if __name__ == "__main__":
    unittest.main()
