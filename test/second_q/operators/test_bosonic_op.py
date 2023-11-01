# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for BosonicOp"""

import unittest
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt, unpack
from qiskit.circuit import Parameter

from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.operators import BosonicOp, PolynomialTensor
import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.operators.commutators import commutator


@ddt
class TestBosonicOp(QiskitNatureTestCase):
    """bosonicOp tests."""

    a = Parameter("a")
    b = Parameter("b")

    op1 = BosonicOp({"+_0 -_0": 1})
    op2 = BosonicOp({"-_0 +_0": 2})
    op3 = BosonicOp({"+_0 -_0": 1, "-_0 +_0": 2})
    op4 = BosonicOp({"+_0 -_0": a})

    def test_neg(self):
        """Test __neg__
        This test method tries to multiply the coefficient by (-1)
        """
        bos_op = -self.op1
        targ = BosonicOp({"+_0 -_0": -1}, num_modes=1)
        self.assertEqual(bos_op, targ)

        bos_op = -self.op4
        targ = BosonicOp({"+_0 -_0": -self.a})
        self.assertEqual(bos_op, targ)

    def test_mul(self):
        """Test __mul__, and __rmul__
        This method tries to multiply from left and right the coefficient
        """
        with self.subTest("rightmul"):
            bos_op = self.op1 * 2
            targ = BosonicOp({"+_0 -_0": 2}, num_modes=1)
            self.assertEqual(bos_op, targ)

            bos_op = self.op1 * self.a
            targ = BosonicOp({"+_0 -_0": self.a})
            self.assertEqual(bos_op, targ)

        with self.subTest("left mul"):
            bos_op = (2 + 1j) * self.op3
            targ = BosonicOp({"+_0 -_0": (2 + 1j), "-_0 +_0": (4 + 2j)}, num_modes=1)
            self.assertEqual(bos_op, targ)

    def test_div(self):
        """Test __truediv__
        This test method tries to divide the coefficient
        """
        bos_op = self.op1 / 2
        targ = BosonicOp({"+_0 -_0": 0.5}, num_modes=1)
        self.assertEqual(bos_op, targ)

        bos_op = self.op1 / self.a
        targ = BosonicOp({"+_0 -_0": 1 / self.a})
        self.assertEqual(bos_op, targ)

    def test_add(self):
        """Test __add__
        This test tries to sum two operators with the same label but different coefficients
        """
        bos_op = self.op1 + self.op2
        targ = self.op3
        self.assertEqual(bos_op, targ)

        bos_op = self.op1 + self.op4
        targ = BosonicOp({"+_0 -_0": 1 + self.a})
        self.assertEqual(bos_op, targ)

    def test_sub(self):
        """Test __sub__
        This test tries to subtract two operators with the same label but different coefficients
        """
        bos_op = self.op3 - self.op2
        targ = BosonicOp({"+_0 -_0": 1, "-_0 +_0": 0}, num_modes=1)
        self.assertEqual(bos_op, targ)

        bos_op = self.op4 - self.op1
        targ = BosonicOp({"+_0 -_0": self.a - 1})
        self.assertEqual(bos_op, targ)

    def test_normal_order(self):
        """test normal_order method
        This test method tries to normal order an BosonicOp, meaning that the resulting label will
        have first all creation operators and then all annihilation operators
        """
        with self.subTest("Test for creation operator"):
            orig = BosonicOp({"+_0": 1}, num_modes=1)
            bos_op = orig.normal_order()
            self.assertEqual(bos_op, orig)

        with self.subTest("Test for annihilation operator"):
            orig = BosonicOp({"-_0": 1}, num_modes=1)
            bos_op = orig.normal_order()
            self.assertEqual(bos_op, orig)

        with self.subTest("Test for number operator"):
            orig = BosonicOp({"+_0 -_0": 1}, num_modes=1)
            bos_op = orig.normal_order()
            self.assertEqual(bos_op, orig)

        with self.subTest("Test for empty operator"):
            orig = BosonicOp({"-_0 +_0": 1}, num_modes=1)
            bos_op = orig.normal_order()
            targ = BosonicOp({"": 1, "+_0 -_0": 1}, num_modes=1)
            self.assertEqual(bos_op, targ)

        with self.subTest("Test for multiple operators 1"):
            orig = BosonicOp({"-_0 +_1": 1}, num_modes=2)
            bos_op = orig.normal_order()
            targ = BosonicOp({"+_1 -_0": 1}, num_modes=2)
            self.assertEqual(bos_op, targ)

        with self.subTest("Test for multiple operators 2"):
            orig = BosonicOp({"-_0 +_0 +_1 -_2": 1}, num_modes=3)
            bos_op = orig.normal_order()
            targ = BosonicOp({"+_1 -_2": 1, "+_0 +_1 -_0 -_2": 1}, num_modes=3)
            self.assertEqual(bos_op, targ)

        with self.subTest("Test normal ordering simplifies"):
            orig = BosonicOp({"-_0 +_1": 2, "+_1 -_0": -1, "+_0": 0.0}, num_modes=2)
            bos_op = orig.normal_order()
            targ = BosonicOp({"+_1 -_0": 1}, num_modes=2)
            self.assertEqual(bos_op, targ)

        with self.subTest("Test parameters"):
            orig = BosonicOp({"-_0 +_0 +_1 -_2": self.a})
            bos_op = orig.normal_order()
            targ = BosonicOp({"+_1 -_2": self.a, "+_0 +_1 -_0 -_2": self.a})
            self.assertEqual(bos_op, targ)

    def test_index_order(self):
        """test index_order method
        This test method tries to index order an BosonicOp, meaning that the resulting label will
        have first all operators acting of the lowest index, and then all operators acting on the
        second lowest index and so on
        """
        with self.subTest("Test for creation operator"):
            orig = BosonicOp({"+_0": 1})
            bos_op = orig.index_order()
            self.assertEqual(bos_op, orig)

        with self.subTest("Test for annihilation operator"):
            orig = BosonicOp({"-_0": 1})
            bos_op = orig.index_order()
            self.assertEqual(bos_op, orig)

        with self.subTest("Test for number operator"):
            orig = BosonicOp({"+_0 -_0": 1})
            bos_op = orig.index_order()
            self.assertEqual(bos_op, orig)

        with self.subTest("Test for empty operator"):
            orig = BosonicOp({"-_0 +_0": 1})
            bos_op = orig.index_order()
            self.assertEqual(bos_op, orig)

        with self.subTest("Test for multiple operators 1"):
            orig = BosonicOp({"+_1 -_0": 1})
            bos_op = orig.index_order()
            targ = BosonicOp({"-_0 +_1": 1})
            self.assertEqual(bos_op, targ)

        with self.subTest("Test for multiple operators 2"):
            orig = BosonicOp({"+_2 -_0 +_1 -_0": 1, "-_0 +_1": 2})
            bos_op = orig.index_order()
            targ = BosonicOp({"-_0 -_0 +_1 +_2": 1, "-_0 +_1": 2})
            self.assertEqual(bos_op, targ)

        with self.subTest("Test index ordering simplifies"):
            orig = BosonicOp({"-_0 +_1": 2, "+_1 -_0": -1, "+_0": 0.0})
            bos_op = orig.index_order()
            targ = BosonicOp({"-_0 +_1": 1})
            self.assertEqual(bos_op, targ)

        with self.subTest("index order + simplify"):
            orig = BosonicOp({"+_1 -_0": 1, "-_0 +_1": 2})
            bos_op = orig.index_order()
            targ = BosonicOp({"-_0 +_1": 3})
            self.assertEqual(bos_op, targ)

    def test_simplify(self):
        """Test simplify
        This test method tries to simplify the operator label
        """
        with self.subTest("simplify does not touch density operators"):
            bos_op = BosonicOp({"+_0 -_0": 1, "-_0 +_0": 1}, num_modes=1)
            simplified_op = bos_op.simplify()
            targ = BosonicOp({"+_0 -_0": 1, "-_0 +_0": 1}, num_modes=1)
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify zero"):
            bos_op = self.op1 - self.op1
            simplified_op = bos_op.simplify()
            targ = BosonicOp.zero()
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify parameters"):
            bos_op = BosonicOp({"+_0 -_0": self.a, "+_0 -_0 +_0": 1j})
            simplified_op = bos_op.simplify()
            targ = BosonicOp({"+_0 -_0": self.a, "+_0 -_0 +_0": 1j})
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify unordered"):
            bos_op = BosonicOp({"+_0 -_0 -_1 +_0": 1})
            simplified_op = bos_op.simplify()
            self.assertEqual(simplified_op, bos_op)

        with self.subTest("simplify commutes with normal_order"):
            bos_op = BosonicOp({"+_0 -_0 +_1 +_0": 1})
            self.assertEqual(bos_op.simplify().normal_order(), bos_op.normal_order().simplify())

        with self.subTest("simplify commutes with index_order"):
            bos_op = BosonicOp({"+_0 -_0 +_1 +_0": 1})
            self.assertEqual(bos_op.simplify().index_order(), bos_op.index_order().simplify())

    def test_commutator(self):
        """Test commutator
        This test method tries the commutation relation between two bosonic operators
        [-_i, +_j] = (-_i +_j) - (+_j -_i) = \\delta_ij
        """
        with self.subTest("commutator same index"):
            bos_op = commutator(BosonicOp({"-_0": 1}), BosonicOp({"+_0": 1}))
            targ = BosonicOp({"": 1})
            self.assertEqual(bos_op, targ)
        with self.subTest("commutator same index reversed"):
            bos_op = commutator(BosonicOp({"+_0": 1}), BosonicOp({"-_0": 1}))
            targ = BosonicOp({"": -1})
            self.assertEqual(bos_op, targ)
        with self.subTest("commutator same different indices"):
            bos_op = commutator(BosonicOp({"+_0": 1}), BosonicOp({"-_1": 1}))
            targ = BosonicOp({})  # 0
            self.assertEqual(bos_op, targ)

    def test_compose(self):
        """Test operator composition
        This test method compares two identical operators.
        One of them is defined directly with the desired label, the other is obtained with a
        composition of two operators
        """
        with self.subTest("single compose"):
            bos_op = BosonicOp({"+_0 -_1": 1}, num_modes=2) @ BosonicOp({"-_0": 1}, num_modes=2)
            targ = BosonicOp({"+_0 -_1 -_0": 1}, num_modes=2)
            self.assertEqual(bos_op, targ)

        with self.subTest("single compose with parameters"):
            bos_op = BosonicOp({"+_0 -_1": self.a}) @ BosonicOp({"-_0": 1})
            targ = BosonicOp({"+_0 -_1 -_0": self.a})
            self.assertEqual(bos_op, targ)

        with self.subTest("multi compose"):
            bos_op = BosonicOp({"+_0 +_1 -_1": 1, "-_0 +_0 -_1": 1}, num_modes=2) @ BosonicOp(
                {"": 1, "-_0 +_1": 1}, num_modes=2
            )
            bos_op = bos_op.simplify()
            targ = BosonicOp(
                {
                    "+_0 +_1 -_1": 1,  # Op1(first term) * Op2(first term)
                    "+_0 +_1 -_1 -_0 +_1": 1,  # Op1(first term) * Op2(second term)
                    "-_0 +_0 -_1": 1,  # Op1(second term) * Op2(first term)
                    "-_0 +_0 -_1 -_0 +_1": 1,  # Op1(second term) * Op2(second term)
                },
                num_modes=2,
            )
            self.assertEqual(bos_op, targ)

        with self.subTest("multi compose with parameters"):
            bos_op = BosonicOp({"+_0 +_1 -_1": self.a, "-_0 +_0 -_1": 1}) @ BosonicOp(
                {"": 1, "-_0 +_1": self.b}
            )
            bos_op = bos_op.simplify()
            targ = BosonicOp(
                {
                    "+_0 +_1 -_1": self.a,  # Op1(first term) * Op2(first term)
                    "+_0 +_1 -_1 -_0 +_1": self.a * self.b,  # Op1(first term) * Op2(second term)
                    "-_0 +_0 -_1": 1,  # Op1(second term) * Op2(first term)
                    "-_0 +_0 -_1 -_0 +_1": self.b,  # Op1(second term) * Op2(second term)
                },
                num_modes=2,
            )
            self.assertEqual(bos_op, targ)

    def test_tensor(self):
        """Test tensor multiplication"""
        bos_op = self.op1.tensor(self.op2)
        targ = BosonicOp({"+_0 -_0 -_1 +_1": 2}, num_modes=2)
        self.assertEqual(bos_op, targ)

        bos_op = self.op4.tensor(self.op2)
        targ = BosonicOp({"+_0 -_0 -_1 +_1": 2 * self.a})
        self.assertEqual(bos_op, targ)

    def test_expand(self):
        """Test reversed tensor multiplication"""
        bos_op = self.op1.expand(self.op2)
        targ = BosonicOp({"-_0 +_0 +_1 -_1": 2}, num_modes=2)
        self.assertEqual(bos_op, targ)

        bos_op = self.op4.expand(self.op2)
        targ = BosonicOp({"-_0 +_0 +_1 -_1": 2 * self.a})
        self.assertEqual(bos_op, targ)

    def test_pow(self):
        """Test __pow__"""
        with self.subTest("square trivial"):
            bos_op = BosonicOp({"+_0 +_1": 3, "+_1 +_0": -1}, num_modes=2) ** 2
            bos_op = bos_op.simplify().normal_order()
            targ = BosonicOp({"+_0 +_0 +_1 +_1": (4 + 0j)}, num_modes=2)
            self.assertEqual(bos_op, targ)

        with self.subTest("square nontrivial"):
            bos_op = BosonicOp({"+_0 +_1 -_1": 3, "+_0 -_0 -_1": 1}, num_modes=2) ** 2
            bos_op = bos_op.normal_order().simplify()
            targ = BosonicOp(
                {
                    "+_0 +_0 +_1 -_1": 9,
                    "+_0 +_0 +_1 +_1 -_1 -_1": 9,  # term1**2
                    "+_0 +_0 +_1 -_0 -_1 -_1": 6,  # term1*term2 + term2*term1.normal_order()
                    "+_0 +_1 -_1 -_1": 3,
                    "+_0 -_1": 3,
                    "+_0 +_0 -_0 -_1": 3,  # term2*term1.normal_order()
                    "+_0 +_0 -_0 -_0 -_1 -_1": 1,
                    "+_0 -_0 -_1 -_1": 1,  # term1**2
                },
                num_modes=2,
            )
            self.assertEqual(bos_op, targ)

        with self.subTest("3rd power"):
            bos_op = (3 * BosonicOp.one()) ** 3
            targ = 27 * BosonicOp.one()
            self.assertEqual(bos_op, targ)

        with self.subTest("0th power"):
            bos_op = BosonicOp({"+_0 +_1 -_1": 3, "-_0 +_0 -_1": 1}, num_modes=2) ** 0
            bos_op = bos_op.simplify()
            targ = BosonicOp.one()
            self.assertEqual(bos_op, targ)

        with self.subTest("square nontrivial with parameters"):
            bos_op = BosonicOp({"+_0 +_1 -_1": self.a, "+_0 -_0 -_1": 1}) ** 2
            bos_op = bos_op.normal_order().simplify()
            targ = BosonicOp(
                {
                    "+_0 +_0 +_1 -_1": self.a * self.a,
                    "+_0 +_0 +_1 +_1 -_1 -_1": self.a * self.a,  # term1**2
                    "+_0 +_0 +_1 -_0 -_1 -_1": 2
                    * self.a,  # term1*term2 + term2*term1.normal_order()
                    "+_0 +_1 -_1 -_1": self.a,
                    "+_0 -_1": self.a,
                    "+_0 +_0 -_0 -_1": self.a,  # term2*term1.normal_order()
                    "+_0 +_0 -_0 -_0 -_1 -_1": 1,
                    "+_0 -_0 -_1 -_1": 1,  # term1**2
                },
                num_modes=2,
            )
            self.assertEqual(bos_op, targ)

    def test_adjoint(self):
        """Test adjoint method"""
        bos_op = BosonicOp(
            {"": 1j, "+_0 +_1 -_1": 3, "+_0 -_0 -_1": 1, "-_0 -_1": 2 + 4j}, num_modes=3
        ).adjoint()
        targ = BosonicOp(
            {"": -1j, "+_1 -_1 -_0": 3, "+_1 +_0 -_0": 1, "+_1 +_0": (2 - 4j)}, num_modes=3
        )
        self.assertEqual(bos_op, targ)

        bos_op = BosonicOp(
            {"": 1j, "+_0 +_1 -_1": 3, "+_0 -_0 -_1": self.a, "-_0 -_1": 2 + 4j}
        ).adjoint()
        targ = BosonicOp(
            {"": -1j, "+_1 -_1 -_0": 3, "+_1 +_0 -_0": self.a.conjugate(), "+_1 +_0": 2 - 4j}
        )
        self.assertEqual(bos_op, targ)

    def test_hermiticity(self):
        """test is_hermitian"""
        with self.subTest("trivial hermitian case"):
            bos_op = BosonicOp({"+_0 -_0": 1}, num_modes=2)
            self.assertTrue(bos_op.is_hermitian())

        with self.subTest("operator hermitian"):
            # deliberately define test operator with duplicate terms in case .adjoint() simplifies terms
            bos_op = (
                1j * BosonicOp({"+_0 -_1 +_2 -_2 -_3 +_3": 1}, num_modes=4)
                + 1j * BosonicOp({"+_0 -_1 +_2 -_2 -_3 +_3": 1}, num_modes=4)
                - 1j * BosonicOp({"-_0 +_1 +_2 -_2 -_3 +_3": 1}, num_modes=4)
                - 1j * BosonicOp({"-_0 +_1 +_2 -_2 -_3 +_3": 1}, num_modes=4)
                + BosonicOp({"+_0 -_1 -_2 +_2 +_3 -_3": 1}, num_modes=4)
                + BosonicOp({"-_0 +_1 -_2 +_2 +_3 -_3": 1}, num_modes=4)
            )
            self.assertTrue(bos_op.is_hermitian())

        with self.subTest("operator not hermitian"):
            bos_op = (
                1j * BosonicOp({"+_0 -_1 +_2 -_2 -_3 +_3": 1}, num_modes=4)
                + 1j * BosonicOp({"+_0 -_1 +_2 -_2 -_3 +_3": 1}, num_modes=4)
                + 1j * BosonicOp({"-_0 +_1 +_2 -_2 -_3 +_3": 1}, num_modes=4)
                + 1j * BosonicOp({"-_0 +_1 +_2 -_2 -_3 +_3": 1}, num_modes=4)
            )
            self.assertFalse(bos_op.is_hermitian())

        with self.subTest("test hermiatin from polynomial tensor"):
            arr1 = np.array([0, 1, 1, 3])
            p_t = PolynomialTensor({"+-": arr1.reshape((2, 2))})
            bos_op = BosonicOp.from_polynomial_tensor(p_t)
            self.assertTrue(bos_op.is_hermitian())

        with self.subTest("test passing atol"):
            bos_op = BosonicOp({"+_0 -_1": 1}, num_modes=2) + (1 + 1e-7) * BosonicOp(
                {"+_1 -_0": 1}, num_modes=2
            )
            self.assertFalse(bos_op.is_hermitian())
            self.assertFalse(bos_op.is_hermitian(atol=1e-8))
            self.assertTrue(bos_op.is_hermitian(atol=1e-6))

        with self.subTest("parameters"):
            bos_op = BosonicOp({"+_0": self.a})
            with self.assertRaisesRegex(ValueError, "parameter"):
                _ = bos_op.is_hermitian()

    def test_equiv(self):
        """test equiv"""
        prev_atol = BosonicOp.atol
        prev_rtol = BosonicOp.rtol
        op3 = self.op1 + (1 + 0.00005) * self.op2
        self.assertFalse(op3.equiv(self.op3))
        BosonicOp.atol = 1e-4
        BosonicOp.rtol = 1e-4
        self.assertTrue(op3.equiv(self.op3))
        BosonicOp.atol = prev_atol
        BosonicOp.rtol = prev_rtol

    def test_induced_norm(self):
        """Test induced norm."""
        op = 3 * BosonicOp({"+_0": 1}, num_modes=1) + 4j * BosonicOp({"-_0": 1}, num_modes=1)
        self.assertAlmostEqual(op.induced_norm(), 7.0)
        self.assertAlmostEqual(op.induced_norm(2), 5.0)

    @unpack
    @data(
        ("", 1, True),  # empty string
        ("+_0", 1, True),  # single term
        ("+_0 -_0", 1, True),  # multiple terms
        ("+_10", 11, True),  # multiple digits
        (" +_0", 1, False),  # leading whitespace
        ("+_0 ", 1, False),  # trailing whitespace
        ("+_0  -_0", 1, False),  # multiple separating spaces
        ("+_0a", 1, False),  # incorrect term pattern
        ("+_a0", 1, False),  # incorrect term pattern
        ("0_+", 1, False),  # incorrect term pattern
        ("something", 1, False),  # incorrect term pattern
        ("+_1", 1, False),  # register length is too short
    )
    def test_validate(self, key: str, length: int, valid: bool):
        """Test key validation."""
        if valid:
            _ = BosonicOp({key: 1.0}, num_modes=length)
        else:
            with self.assertRaises(QiskitNatureError):
                _ = BosonicOp({key: 1.0}, num_modes=length)

    def test_from_polynomial_tensor(self):
        """Test from PolynomialTensor construction"""

        with self.subTest("dense tensor"):
            r_l = 2
            p_t = PolynomialTensor(
                {
                    "+-": np.arange(1, 5).reshape((r_l, r_l)),
                    "++--": np.arange(1, 17).reshape((r_l, r_l, r_l, r_l)),
                }
            )
            op = BosonicOp.from_polynomial_tensor(p_t)

            expected = BosonicOp(
                {
                    "+_0 -_0": 1,
                    "+_0 -_1": 2,
                    "+_1 -_0": 3,
                    "+_1 -_1": 4,
                    "+_0 +_0 -_0 -_0": 1,
                    "+_0 +_0 -_0 -_1": 2,
                    "+_0 +_0 -_1 -_0": 3,
                    "+_0 +_0 -_1 -_1": 4,
                    "+_0 +_1 -_0 -_0": 5,
                    "+_0 +_1 -_0 -_1": 6,
                    "+_0 +_1 -_1 -_0": 7,
                    "+_0 +_1 -_1 -_1": 8,
                    "+_1 +_0 -_0 -_0": 9,
                    "+_1 +_0 -_0 -_1": 10,
                    "+_1 +_0 -_1 -_0": 11,
                    "+_1 +_0 -_1 -_1": 12,
                    "+_1 +_1 -_0 -_0": 13,
                    "+_1 +_1 -_0 -_1": 14,
                    "+_1 +_1 -_1 -_0": 15,
                    "+_1 +_1 -_1 -_1": 16,
                },
                num_modes=r_l,
            )

            self.assertEqual(op, expected)

        if _optionals.HAS_SPARSE:
            import sparse as sp  # pylint: disable=import-error

            with self.subTest("sparse tensor"):
                r_l = 2
                p_t = PolynomialTensor(
                    {
                        "+-": sp.as_coo({(0, 0): 1, (1, 0): 2}, shape=(r_l, r_l)),
                        "++--": sp.as_coo(
                            {(0, 0, 0, 1): 1, (1, 0, 1, 1): 2}, shape=(r_l, r_l, r_l, r_l)
                        ),
                    }
                )
                op = BosonicOp.from_polynomial_tensor(p_t)

                expected = BosonicOp(
                    {
                        "+_0 -_0": 1,
                        "+_1 -_0": 2,
                        "+_0 +_0 -_0 -_1": 1,
                        "+_1 +_0 -_1 -_1": 2,
                    },
                    num_modes=r_l,
                )

                self.assertEqual(op, expected)

        with self.subTest("compose operation order"):
            r_l = 2
            p_t = PolynomialTensor(
                {
                    "+-": np.arange(1, 5).reshape((r_l, r_l)),
                    "++--": np.arange(1, 17).reshape((r_l, r_l, r_l, r_l)),
                }
            )
            op = BosonicOp.from_polynomial_tensor(p_t)

            a = op @ op
            b = BosonicOp.from_polynomial_tensor(p_t @ p_t)
            self.assertEqual(a, b)

        with self.subTest("tensor operation order"):
            r_l = 2
            p_t = PolynomialTensor(
                {
                    "+-": np.arange(1, 5).reshape((r_l, r_l)),
                    "++--": np.arange(1, 17).reshape((r_l, r_l, r_l, r_l)),
                }
            )
            op = BosonicOp.from_polynomial_tensor(p_t)

            self.assertEqual(op ^ op, BosonicOp.from_polynomial_tensor(p_t ^ p_t))

    def test_no_num_modes(self):
        """Test operators with automatic register length"""
        op0 = BosonicOp({"": 1})
        op1 = BosonicOp({"+_0 -_0": 1})
        op2 = BosonicOp({"-_0 +_1": 2})

        with self.subTest("Inferred register length"):
            self.assertEqual(op0.num_modes, 0)
            self.assertEqual(op1.num_modes, 1)
            self.assertEqual(op2.num_modes, 2)

        with self.subTest("Mathematical operations"):
            self.assertEqual((op0 + op2).num_modes, 2)
            self.assertEqual((op1 + op2).num_modes, 2)
            self.assertEqual((op0 @ op2).num_modes, 2)
            self.assertEqual((op1 @ op2).num_modes, 2)
            self.assertEqual((op1 ^ op2).num_modes, 3)

        with self.subTest("Equality"):
            op3 = BosonicOp({"+_0 -_0": 1}, num_modes=3)
            self.assertEqual(op1, op3)
            self.assertTrue(op1.equiv(1.000001 * op3))

    def test_terms(self):
        """Test terms generator."""
        op = BosonicOp(
            {
                "+_0": 1,
                "-_0 +_1": 2,
                "+_1 -_1 +_2": 2,
            }
        )

        terms = [([("+", 0)], 1), ([("-", 0), ("+", 1)], 2), ([("+", 1), ("-", 1), ("+", 2)], 2)]

        with self.subTest("terms"):
            self.assertEqual(list(op.terms()), terms)

        with self.subTest("from_terms"):
            self.assertEqual(BosonicOp.from_terms(terms), op)

    def test_permute_indices(self):
        """Test index permutation method."""
        op = BosonicOp(
            {
                "+_0 -_1": 1,
                "+_1 -_2": 2,
            },
            num_modes=4,
        )

        with self.subTest("wrong permutation length"):
            with self.assertRaises(ValueError):
                _ = op.permute_indices([1, 0])

        with self.subTest("actual permutation"):
            permuted_op = op.permute_indices([2, 1, 3, 0])

            self.assertEqual(permuted_op, BosonicOp({"+_2 -_1": 1, "+_1 -_3": 2}, num_modes=4))

    def test_reg_len_with_skipped_key_validation(self):
        """Test the behavior of `register_length` after key validation was skipped."""
        new_op = BosonicOp({"+_0 -_1": 1}, validate=False)
        self.assertIsNone(new_op.num_modes)
        self.assertEqual(new_op.register_length, 2)


if __name__ == "__main__":
    unittest.main()
