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

import numpy as np
from ddt import data, ddt

from qiskit_nature.second_q.operators import VibrationalOp


@ddt
class TestVibrationalOp(QiskitNatureTestCase):
    """VibrationalOp tests."""

    # op1 = VibrationalOp({"+_0_0 -_0_0": 1}, num_modes=1, num_modals=1)
    op1 = VibrationalOp({"+_0_0 -_0_0": 1})

    op2 = VibrationalOp({"-_0_0 +_0_0": 2})
    op3 = VibrationalOp({"+_0_0 -_0_0": 1, "-_0_0 +_0_0": 2})

    # def test_no_num_spin_orbitals(self):
    #     """Test operators with automatic register length"""
    #     op0 = FermionicOp({"": 1})
    #     op1 = FermionicOp({"+_0 -_0": 1})
    #     op2 = FermionicOp({"-_0 +_1": 2})

    #     with self.subTest("Inferred register length"):
    #         self.assertEqual(op0.num_spin_orbitals, 0)
    #         self.assertEqual(op1.num_spin_orbitals, 1)
    #         self.assertEqual(op2.num_spin_orbitals, 2)

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
        targ = VibrationalOp({"+_0_0 -_0_0": 1, "-_0_0 +_0_0": 0}, num_modes=1, num_modals=1)
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
            targ = VibrationalOp(
                {
                    "+_0_0 +_1_0 -_1_0 -_0_0 +_1_0": 1,
                    "-_0_0 +_0_0 -_1_0 -_0_0 +_1_0": 1,
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

    # def test_pow(self):
    #     """Test __pow__"""
    #     with self.subTest("square trivial"):
    #         fer_op = FermionicOp({"+_0 +_1 -_1": 3, "-_0 +_0 -_1": 1}, num_spin_orbitals=2) ** 2
    #         fer_op = fer_op.simplify()
    #         targ = FermionicOp.zero()
    #         self.assertEqual(fer_op, targ)

    #     with self.subTest("square nontrivial"):
    #         fer_op = FermionicOp({"+_0 +_1 -_1": 3, "+_0 -_0 -_1": 1}, num_spin_orbitals=2) ** 2
    #         fer_op = fer_op.simplify()
    #         targ = FermionicOp({"+_0 -_1": -3}, num_spin_orbitals=2)
    #         self.assertEqual(fer_op, targ)

    #     with self.subTest("3rd power"):
    #         fer_op = (3 * FermionicOp.one()) ** 3
    #         targ = 27 * FermionicOp.one()
    #         self.assertEqual(fer_op, targ)

    #     with self.subTest("0th power"):
    #         fer_op = FermionicOp({"+_0 +_1 -_1": 3, "-_0 +_0 -_1": 1}, num_spin_orbitals=2) ** 0
    #         fer_op = fer_op.simplify()
    #         targ = FermionicOp.one()
    #         self.assertEqual(fer_op, targ)

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


#     def test_simplify(self):
#         """Test simplify"""
#         with self.subTest("simplify integer"):
#             fer_op = FermionicOp({"+_0 -_0": 1, "+_0 -_0 +_0 -_0": 1}, num_spin_orbitals=1)
#             simplified_op = fer_op.simplify()
#             targ = FermionicOp({"+_0 -_0": 2}, num_spin_orbitals=1)
#             self.assertEqual(simplified_op, targ)

#         with self.subTest("simplify complex"):
#             fer_op = FermionicOp({"+_0 -_0": 1, "+_0 -_0 +_0 -_0": 1j}, num_spin_orbitals=1)
#             simplified_op = fer_op.simplify()
#             targ = FermionicOp({"+_0 -_0": 1 + 1j}, num_spin_orbitals=1)
#             self.assertEqual(simplified_op, targ)

#         with self.subTest("simplify doesn't reorder"):
#             fer_op = FermionicOp({"-_0 +_1": 1 + 0j}, num_spin_orbitals=2)
#             simplified_op = fer_op.simplify()
#             self.assertEqual(simplified_op, fer_op)

#             fer_op = FermionicOp({"-_1 +_0": 1 + 0j}, num_spin_orbitals=2)
#             simplified_op = fer_op.simplify()
#             self.assertEqual(simplified_op, fer_op)

#         with self.subTest("simplify zero"):
#             fer_op = self.op1 - self.op1
#             simplified_op = fer_op.simplify()
#             targ = FermionicOp.zero()
#             self.assertEqual(simplified_op, targ)

#         with self.subTest("simplify commutes with normal_order"):
#             fer_op = FermionicOp({"-_0 +_1": 1}, num_spin_orbitals=2)
#             self.assertEqual(fer_op.simplify().normal_order(), fer_op.normal_order().simplify())

#         with self.subTest("simplify + index order"):
#             orig = FermionicOp({"+_1 -_0 +_0 -_0": 1, "-_0 +_1": 2})
#             fer_op = orig.simplify().index_order()
#             targ = FermionicOp({"-_0 +_1": 1})
#             self.assertEqual(fer_op, targ)

#     def test_hermiticity(self):
#         """test is_hermitian"""
#         with self.subTest("operator hermitian"):
#             # deliberately define test operator with duplicate terms in case .adjoint() simplifies terms
#             fer_op = (
#                 1j * FermionicOp({"+_0 -_1 +_2 -_2 -_3 +_3": 1}, num_spin_orbitals=4)
#                 + 1j * FermionicOp({"+_0 -_1 +_2 -_2 -_3 +_3": 1}, num_spin_orbitals=4)
#                 + 1j * FermionicOp({"-_0 +_1 +_2 -_2 -_3 +_3": 1}, num_spin_orbitals=4)
#                 + 1j * FermionicOp({"-_0 +_1 +_2 -_2 -_3 +_3": 1}, num_spin_orbitals=4)
#                 + FermionicOp({"+_0 -_1 -_2 +_2 +_3 -_3": 1}, num_spin_orbitals=4)
#                 - FermionicOp({"-_0 +_1 -_2 +_2 +_3 -_3": 1}, num_spin_orbitals=4)
#             )
#             self.assertTrue(fer_op.is_hermitian())

#         with self.subTest("operator not hermitian"):
#             fer_op = (
#                 1j * FermionicOp({"+_0 -_1 +_2 -_2 -_3 +_3": 1}, num_spin_orbitals=4)
#                 + 1j * FermionicOp({"+_0 -_1 +_2 -_2 -_3 +_3": 1}, num_spin_orbitals=4)
#                 - 1j * FermionicOp({"-_0 +_1 +_2 -_2 -_3 +_3": 1}, num_spin_orbitals=4)
#                 - 1j * FermionicOp({"-_0 +_1 +_2 -_2 -_3 +_3": 1}, num_spin_orbitals=4)
#             )
#             self.assertFalse(fer_op.is_hermitian())

#         with self.subTest("test require normal order"):
#             fer_op = (
#                 FermionicOp({"+_0 -_0 -_1": 1}, num_spin_orbitals=2)
#                 - FermionicOp({"+_1 -_0 +_0": 1}, num_spin_orbitals=2)
#                 + FermionicOp({"+_1": 1}, num_spin_orbitals=2)
#             )
#             self.assertTrue(fer_op.is_hermitian())

#         with self.subTest("test passing atol"):
#             fer_op = FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2) + (1 + 1e-7) * FermionicOp(
#                 {"+_1 -_0": 1}, num_spin_orbitals=2
#             )
#             self.assertFalse(fer_op.is_hermitian())
#             self.assertFalse(fer_op.is_hermitian(atol=1e-8))
#             self.assertTrue(fer_op.is_hermitian(atol=1e-6))

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

#     def test_to_matrix(self):
#         """Test to_matrix"""
#         with self.subTest("identity operator matrix"):
#             op = FermionicOp.one()
#             op.num_spin_orbitals = 2
#             mat = op.to_matrix(sparse=False)
#             targ = np.eye(4)
#             self.assertTrue(np.allclose(mat, targ))

#         with self.subTest("number operator matrix"):
#             mat = FermionicOp({"+_1 -_1": 1}, num_spin_orbitals=2).to_matrix(sparse=False)
#             targ = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
#             self.assertTrue(np.allclose(mat, targ))

#         with self.subTest("emptiness operator matrix"):
#             mat = FermionicOp({"-_1 +_1": 1}, num_spin_orbitals=2).to_matrix(sparse=False)
#             targ = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
#             self.assertTrue(np.allclose(mat, targ))

#         with self.subTest("raising operator matrix"):
#             mat = FermionicOp({"+_1": 1}, num_spin_orbitals=2).to_matrix(sparse=False)
#             targ = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0]])
#             self.assertTrue(np.allclose(mat, targ))

#         with self.subTest("lowering operator matrix"):
#             mat = FermionicOp({"-_1": 1}, num_spin_orbitals=2).to_matrix(sparse=False)
#             targ = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0]])
#             self.assertTrue(np.allclose(mat, targ))

#         with self.subTest("nontrivial sparse matrix"):
#             mat = FermionicOp(
#                 {"-_0 +_0 +_1 -_1 +_3": 3j, "-_0 +_1 -_1 +_2 -_3": -2}, num_spin_orbitals=4
#             ).to_matrix()
#             targ = csc_matrix(([-3j, 3j, -2], ([5, 7, 6], [4, 6, 13])), shape=(16, 16))
#             self.assertTrue((mat != targ).nnz == 0)

#         with self.subTest("Test Hydrogen spectrum"):
#             h2_labels = {
#                 "+_0 -_1 +_2 -_3": 0.18093120148374142,
#                 "+_0 -_1 -_2 +_3": -0.18093120148374134,
#                 "-_0 +_1 +_2 -_3": -0.18093120148374134,
#                 "-_0 +_1 -_2 +_3": 0.18093120148374128,
#                 "+_3 -_3": -0.4718960038869427,
#                 "+_2 -_2": -1.2563391028292563,
#                 "+_2 -_2 +_3 -_3": 0.48365053378098793,
#                 "+_1 -_1": -0.4718960038869427,
#                 "+_1 -_1 +_3 -_3": 0.6985737398458793,
#                 "+_1 -_1 +_2 -_2": 0.6645817352647293,
#                 "+_0 -_0": -1.2563391028292563,
#                 "+_0 -_0 +_3 -_3": 0.6645817352647293,
#                 "+_0 -_0 +_2 -_2": 0.6757101625347564,
#                 "+_0 -_0 +_1 -_1": 0.48365053378098793,
#             }
#             h2_matrix = FermionicOp(h2_labels, num_spin_orbitals=4).to_matrix()
#             evals, evecs = eigs(h2_matrix)
#             self.assertTrue(np.isclose(np.min(evals), -1.8572750))
#             # make sure the ground state has support only in the 2-particle subspace
#             groundstate = evecs[:, np.argmin(evals)]
#             for idx in np.where(~np.isclose(groundstate, 0))[0]:
#                 binary = f"{idx:0{4}b}"
#                 self.assertEqual(binary.count("1"), 2)

#     def test_normal_order(self):
#         """test normal_order method"""
#         with self.subTest("Test for creation operator"):
#             orig = FermionicOp({"+_0": 1}, num_spin_orbitals=1)
#             fer_op = orig.normal_order()
#             self.assertEqual(fer_op, orig)

#         with self.subTest("Test for annihilation operator"):
#             orig = FermionicOp({"-_0": 1}, num_spin_orbitals=1)
#             fer_op = orig.normal_order()
#             self.assertEqual(fer_op, orig)

#         with self.subTest("Test for number operator"):
#             orig = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=1)
#             fer_op = orig.normal_order()
#             self.assertEqual(fer_op, orig)

#         with self.subTest("Test for empty operator"):
#             orig = FermionicOp({"-_0 +_0": 1}, num_spin_orbitals=1)
#             fer_op = orig.normal_order()
#             targ = FermionicOp({"": 1, "+_0 -_0": -1}, num_spin_orbitals=1)
#             self.assertEqual(fer_op, targ)

#         with self.subTest("Test for multiple operators 1"):
#             orig = FermionicOp({"-_0 +_1": 1}, num_spin_orbitals=2)
#             fer_op = orig.normal_order()
#             targ = FermionicOp({"+_1 -_0": -1}, num_spin_orbitals=2)
#             self.assertEqual(fer_op, targ)

#         with self.subTest("Test for multiple operators 2"):
#             orig = FermionicOp({"-_0 +_0 +_1 -_2": 1}, num_spin_orbitals=3)
#             fer_op = orig.normal_order()
#             targ = FermionicOp({"+_1 -_2": 1, "+_0 +_1 -_0 -_2": 1}, num_spin_orbitals=3)
#             self.assertEqual(fer_op, targ)

#         with self.subTest("Test normal ordering simplifies"):
#             orig = FermionicOp({"-_0 +_1": 1, "+_1 -_0": -1, "+_0": 0.0}, num_spin_orbitals=2)
#             fer_op = orig.normal_order()
#             targ = FermionicOp({"+_1 -_0": -2}, num_spin_orbitals=2)
#             self.assertEqual(fer_op, targ)

#     def test_index_order(self):
#         """test index_order method"""
#         with self.subTest("Test for creation operator"):
#             orig = FermionicOp({"+_0": 1})
#             fer_op = orig.index_order()
#             self.assertEqual(fer_op, orig)

#         with self.subTest("Test for annihilation operator"):
#             orig = FermionicOp({"-_0": 1})
#             fer_op = orig.index_order()
#             self.assertEqual(fer_op, orig)

#         with self.subTest("Test for number operator"):
#             orig = FermionicOp({"+_0 -_0": 1})
#             fer_op = orig.index_order()
#             self.assertEqual(fer_op, orig)

#         with self.subTest("Test for empty operator"):
#             orig = FermionicOp({"-_0 +_0": 1})
#             fer_op = orig.index_order()
#             self.assertEqual(fer_op, orig)

#         with self.subTest("Test for multiple operators 1"):
#             orig = FermionicOp({"+_1 -_0": 1})
#             fer_op = orig.index_order()
#             targ = FermionicOp({"-_0 +_1": -1})
#             self.assertEqual(fer_op, targ)

#         with self.subTest("Test for multiple operators 2"):
#             orig = FermionicOp({"+_2 -_0 +_1 -_0": 1, "-_0 +_1": 2})
#             fer_op = orig.index_order()
#             targ = FermionicOp({"-_0 -_0 +_1 +_2": 1, "-_0 +_1": 2})
#             self.assertEqual(fer_op, targ)

#         with self.subTest("Test index ordering simplifies"):
#             orig = FermionicOp({"-_0 +_1": 1, "+_1 -_0": -1, "+_0": 0.0})
#             fer_op = orig.index_order()
#             targ = FermionicOp({"-_0 +_1": 2})
#             self.assertEqual(fer_op, targ)

#         with self.subTest("index order + simplify"):
#             orig = FermionicOp({"+_1 -_0 +_0 -_0": 1, "-_0 +_1": 2})
#             fer_op = orig.index_order().simplify()
#             targ = FermionicOp({"-_0 +_1": 1})
#             self.assertEqual(fer_op, targ)

#     def test_induced_norm(self):
#         """Test induced norm."""
#         op = 3 * FermionicOp({"+_0": 1}, num_spin_orbitals=1) + 4j * FermionicOp(
#             {"-_0": 1}, num_spin_orbitals=1
#         )
#         self.assertAlmostEqual(op.induced_norm(), 7.0)
#         self.assertAlmostEqual(op.induced_norm(2), 5.0)

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
#     def test_validate(self, key: str, length: int, valid: bool):
#         """Test key validation."""
#         if valid:
#             _ = FermionicOp({key: 1.0}, num_spin_orbitals=length)
#         else:
#             with self.assertRaises(QiskitNatureError):
#                 _ = FermionicOp({key: 1.0}, num_spin_orbitals=length)

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
