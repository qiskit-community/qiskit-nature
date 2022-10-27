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

    #     with self.subTest("to_matrix"):
    #         ref = np.array([[0, 0], [0, 1]])
    #         np.testing.assert_array_almost_equal(op1.to_matrix(False), ref)
    #         op1.num_spin_orbitals = 2
    #         np.testing.assert_array_almost_equal(op1.to_matrix(False), np.kron(ref, np.eye(2)))

    #         ref = np.array([[1]])
    #         np.testing.assert_array_almost_equal(op0.to_matrix(False), ref)

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


#     @unpack
#     @data(
#         ("", 1, True),  # empty string
#         ("+_0", 1, True),  # single term
#         ("+_0 -_0", 1, True),  # multiple terms
#         ("+_10", 11, True),  # multiple digits
#         (" +_0", 1, False),  # leading whitespace
#         ("+_0 ", 1, False),  # trailing whitespace
#         ("+_0  -_0", 1, False),  # multiple separating spaces
#         ("+_0a", 1, False),  # incorrect term pattern
#         ("+_a0", 1, False),  # incorrect term pattern
#         ("0_+", 1, False),  # incorrect term pattern
#         ("something", 1, False),  # incorrect term pattern
#         ("+_1", 1, False),  # register length is too short
#     )
# def test_validate(self, key: str, length: int, valid: bool):
#     """Test key validation."""
#     if valid:
#         _ = FermionicOp({key: 1.0}, num_spin_orbitals=length)
#     else:
#         with self.assertRaises(QiskitNatureError):
#             _ = FermionicOp({key: 1.0}, num_spin_orbitals=length)

#     def test_from_polynomial_tensor(self):
#         """Test from PolynomialTensor construction"""

#         with self.subTest("dense tensor"):
#             r_l = 2
#             p_t = PolynomialTensor(
#                 {
#                     "+-": np.arange(1, 5).reshape((r_l, r_l)),
#                     "++--": np.arange(1, 17).reshape((r_l, r_l, r_l, r_l)),
#                 }
#             )
#             op = FermionicOp.from_polynomial_tensor(p_t)

#             expected = FermionicOp(
#                 {
#                     "+_0 -_0": 1,
#                     "+_0 -_1": 2,
#                     "+_1 -_0": 3,
#                     "+_1 -_1": 4,
#                     "+_0 +_0 -_0 -_0": 1,
#                     "+_0 +_0 -_0 -_1": 2,
#                     "+_0 +_0 -_1 -_0": 3,
#                     "+_0 +_0 -_1 -_1": 4,
#                     "+_0 +_1 -_0 -_0": 5,
#                     "+_0 +_1 -_0 -_1": 6,
#                     "+_0 +_1 -_1 -_0": 7,
#                     "+_0 +_1 -_1 -_1": 8,
#                     "+_1 +_0 -_0 -_0": 9,
#                     "+_1 +_0 -_0 -_1": 10,
#                     "+_1 +_0 -_1 -_0": 11,
#                     "+_1 +_0 -_1 -_1": 12,
#                     "+_1 +_1 -_0 -_0": 13,
#                     "+_1 +_1 -_0 -_1": 14,
#                     "+_1 +_1 -_1 -_0": 15,
#                     "+_1 +_1 -_1 -_1": 16,
#                 },
#                 num_spin_orbitals=r_l,
#             )

#             self.assertEqual(op, expected)

#         if _optionals.HAS_SPARSE:
#             import sparse as sp  # pylint: disable=import-error

#             with self.subTest("sparse tensor"):
#                 r_l = 2
#                 p_t = PolynomialTensor(
#                     {
#                         "+-": sp.as_coo({(0, 0): 1, (1, 0): 2}, shape=(r_l, r_l)),
#                         "++--": sp.as_coo(
#                             {(0, 0, 0, 1): 1, (1, 0, 1, 1): 2}, shape=(r_l, r_l, r_l, r_l)
#                         ),
#                     }
#                 )
#                 op = FermionicOp.from_polynomial_tensor(p_t)

#                 expected = FermionicOp(
#                     {
#                         "+_0 -_0": 1,
#                         "+_1 -_0": 2,
#                         "+_0 +_0 -_0 -_1": 1,
#                         "+_1 +_0 -_1 -_1": 2,
#                     },
#                     num_spin_orbitals=r_l,
#                 )

#                 self.assertEqual(op, expected)

#         with self.subTest("compose operation order"):
#             r_l = 2
#             p_t = PolynomialTensor(
#                 {
#                     "+-": np.arange(1, 5).reshape((r_l, r_l)),
#                     "++--": np.arange(1, 17).reshape((r_l, r_l, r_l, r_l)),
#                 }
#             )
#             op = FermionicOp.from_polynomial_tensor(p_t)

#             self.assertEqual(op @ op, FermionicOp.from_polynomial_tensor(p_t @ p_t))

#         with self.subTest("tensor operation order"):
#             r_l = 2
#             p_t = PolynomialTensor(
#                 {
#                     "+-": np.arange(1, 5).reshape((r_l, r_l)),
#                     "++--": np.arange(1, 17).reshape((r_l, r_l, r_l, r_l)),
#                 }
#             )
#             op = FermionicOp.from_polynomial_tensor(p_t)

#             self.assertEqual(op ^ op, FermionicOp.from_polynomial_tensor(p_t ^ p_t))


# @ddt
# class TestVibrationalOp(QiskitNatureTestCase):
#     """VibrationalOp tests."""

# def setUp(self):
#     super().setUp()
#     self.labels = [("+_1*0 -_1*1", 1215.375), ("+_2*0 -_2*1 +_3*0 -_3*0", -6.385)]
#     self.labels_double = [
#         ("+_1*0 -_1*1", 2 * 1215.375),
#         ("+_2*0 -_2*1 +_3*0 -_3*0", -2 * 6.385),
#     ]
#     self.labels_divided_3 = [
#         ("+_1*0 -_1*1", 1215.375 / 3),
#         ("+_2*0 -_2*1 +_3*0 -_3*0", -6.385 / 3),
#     ]
#     self.labels_neg = [
#         ("+_1*0 -_1*1", -1215.375),
#         ("+_2*0 -_2*1 +_3*0 -_3*0", 6.385),
#     ]
#     self.vibr_spin_op = VibrationalOp(self.labels, 4, 2)

# def assertSpinEqual(self, first: VibrationalOp, second: VibrationalOp):
#     """Fail if two VibrationalOps have different matrix representations."""
#     self.assertEqual(first._labels, second._labels)
#     np.testing.assert_array_almost_equal(first._coeffs, second._coeffs)

# def test_init_pm_label(self):
#     """Test __init__ with plus and minus label"""
#     with self.subTest("minus plus"):
#         result = VibrationalOp({"+_0_0 -_0_1": 2}, 1, 2)
#         desired = [("+-", (2 + 0j))]
#         self.assertEqual(result.to_list(), desired)

#     with self.subTest("plus minus"):
#         result = VibrationalOp([("-_0*0 +_0*1", 2)], 1, 2)
#         desired = [("-+", (2 + 0j))]
#         self.assertEqual(result.to_list(), desired)

#     with self.subTest("plus minus minus plus"):
#         result = VibrationalOp([("+_0*0 -_0*1 -_1*0 +_1*1", 3)], 2, 2)
#         desired = [("+--+", (3 + 0j))]

#         # Note: the order of list is irrelevant.
#         self.assertSetEqual(frozenset(result.to_list()), frozenset(desired))

# @data("X_0*0 +_0*0")
# def test_init_invalid_label(self, label):
#     """Test __init__ for invalid label"""
#     with self.assertRaises(ValueError):
#         VibrationalOp(label, 1, 1)


# def test_hermiticity(self):
#     """test is_hermitian"""
#     with self.subTest("operator hermitian"):
#         # deliberately define test operator with duplicate terms in case .adjoint() simplifies terms
#         test_op = (
#             1j * VibrationalOp("+-", 2, 1)
#             + 1j * VibrationalOp("+-", 2, 1)
#             - 1j * VibrationalOp("-+", 2, 1)
#             - 1j * VibrationalOp("-+", 2, 1)
#         )
#         self.assertTrue(test_op.is_hermitian())

#     with self.subTest("operator not hermitian"):
#         test_op = (
#             1j * VibrationalOp("+-", 2, 1)
#             + 1j * VibrationalOp("+-", 2, 1)
#             - 1j * VibrationalOp("-+", 2, 1)
#         )
#         self.assertFalse(test_op.is_hermitian())

#     with self.subTest("test passing atol"):
#         test_op = 1j * VibrationalOp("+-", 2, 1) - (1 + 1e-7) * 1j * VibrationalOp("-+", 2, 1)
#         self.assertFalse(test_op.is_hermitian())
#         self.assertFalse(test_op.is_hermitian(atol=1e-8))
#         self.assertTrue(test_op.is_hermitian(atol=1e-6))

# def test_simplify(self):
#     """Test simplify"""
#     test_op = (
#         1j * VibrationalOp("+-", 2, 1)
#         + 1j * VibrationalOp("+-", 2, 1)
#         - 1j * VibrationalOp("-+", 2, 1)
#         - 1j * VibrationalOp("-+", 2, 1)
#     )
#     expected = [("+-", 2j), ("-+", -2j)]
#     self.assertEqual(test_op.simplify().to_list(), expected)

# def test_equiv(self):
#     """test equiv"""
#     op1 = VibrationalOp("+-", 2, 1) + VibrationalOp("-+", 2, 1)
#     op2 = VibrationalOp("+-", 2, 1)
#     op3 = VibrationalOp("+-", 2, 1) + (1 + 1e-7) * VibrationalOp("-+", 2, 1)
#     self.assertFalse(op1.equiv(op2))
#     self.assertFalse(op1.equiv(op3))
#     self.assertTrue(op1.equiv(op3, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
