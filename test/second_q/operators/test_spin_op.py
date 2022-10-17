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

"""Test for SpinOp"""

import unittest
from fractions import Fraction
from functools import lru_cache
from itertools import product
from test import QiskitNatureTestCase
from test.second_q.operators.utils import str2list, str2str, str2tuple

import numpy as np
from ddt import data, ddt, unpack
from qiskit.quantum_info import Pauli

from qiskit_nature.second_q.operators import SpinOp, FermionicOp
import unittest
from test import QiskitNatureTestCase
from ddt import ddt, data, unpack

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs

from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.operators import FermionicOp, PolynomialTensor
import qiskit_nature.optionals as _optionals



@ddt
class TestSpinOp(QiskitNatureTestCase):
    """SpinOp tests."""

    op1 = SpinOp({"X_0 Y_0": 1}, num_orbitals=1)
    op2 = SpinOp({"X_0 Z_0": 2}, num_orbitals=1)
    op3 = SpinOp({"X_0 Y_0": 1, "X_0 Z_0": 2}, num_orbitals=1)

    def test_neg(self):
        """Test __neg__"""
        spin_op = -self.op1
        targ = SpinOp({"X_0 Y_0": -1}, num_orbitals=1)
        self.assertEqual(spin_op, targ)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        with self.subTest("rightmul"):
            spin_op = self.op1 * 2
            targ = SpinOp({"X_0 Y_0": 2}, num_orbitals=1)
            self.assertEqual(spin_op, targ)

        with self.subTest("left mul"):
            spin_op = (2 + 1j) * self.op3
            targ = SpinOp({"X_0 Y_0": (2 + 1j), "X_0 Z_0": (4 + 2j)}, num_orbitals=1)
            self.assertEqual(spin_op, targ)

    def test_div(self):
        """Test __truediv__"""
        spin_op = self.op1 / 2
        targ = SpinOp({"X_0 Y_0": 0.5}, num_orbitals=1)
        self.assertEqual(spin_op, targ)

    def test_add(self):
        """Test __add__"""
        spin_op = self.op1 + self.op2
        targ = self.op3
        self.assertEqual(spin_op, targ)

    def test_sub(self):
        """Test __sub__"""
        spin_op = self.op3 - self.op2
        targ = SpinOp({"X_0 Y_0": 1, "X_0 Z_0": 0}, num_orbitals=1)
        self.assertEqual(spin_op, targ)

    def test_tensor(self):
        """Test tensor multiplication"""
        spin_op = self.op1.tensor(self.op2)
        targ = SpinOp({"X_0 Y_0 X_1 Z_1": 2}, num_orbitals=2)
        self.assertEqual(spin_op, targ)

    def test_expand(self):
        """Test reversed tensor multiplication"""
        spin_op = self.op1.expand(self.op2)
        targ = SpinOp({"X_0 Z_0 X_1 Y_1": 2}, num_orbitals=2)
        self.assertEqual(spin_op, targ)

    # def test_pow(self):
    #     """Test __pow__"""
    #     # TODO
    #     with self.subTest("square trivial"):
    #         spin_op = SpinOp({"X_0 X_1": 3, "X_0 X_1": -3}, num_orbitals=2) ** 2
    #         spin_op = spin_op.simplify()
    #         targ = SpinOp.zero()
    #         self.assertEqual(spin_op, targ)
    #
    #     with self.subTest("square nontrivial"):
    #         spin_op = SpinOp({"X_0 X_1 Y_1": 3, "X_0 Y_0 Y_1": 1}, num_orbitals=2) ** 2
    #         spin_op = spin_op.simplify()
    #         targ = SpinOp({"Y_0 X_1": 6}, num_orbitals=2)
    #         self.assertEqual(spin_op, targ)
    #
    #     with self.subTest("3rd power"):
    #         spin_op = (3 * SpinOp.one()) ** 3
    #         targ = 27 * SpinOp.one()
    #         self.assertEqual(spin_op, targ)
    #
    #     with self.subTest("0th power"):
    #         spin_op = SpinOp({"X_0 X_1 Y_1": 3, "Y_0 X_0 Y_1": 1}, num_orbitals=2) ** 0
    #         spin_op = spin_op.simplify()
    #         targ = SpinOp.one()
    #         self.assertEqual(spin_op, targ)

    def test_adjoint(self):
        """Test adjoint method"""
        spin_op = SpinOp(
            {"": 1j, "X_0 Y_1 X_1": 3, "X_0 Y_0 Y_1": 1, "Y_0 Y_1": 2 + 4j}, num_orbitals=3
        ).adjoint()

        targ = SpinOp(
            {"": -1j, "X_0 Y_1 X_1": -3, 'X_1 Y_1 X_0': 1, 'Y_1 Y_0 X_0': 2 - 4j}, num_orbitals=3
        )
        self.assertEqual(spin_op, targ)

    # def test_simplify(self):
    #     """Test simplify"""
    #     with self.subTest("simplify integer"):
    #         fer_op = FermionicOp({"+_0 -_0": 1, "+_0 -_0 +_0 -_0": 1}, num_spin_orbitals=1)
    #         simplified_op = fer_op.simplify()
    #         targ = FermionicOp({"+_0 -_0": 2}, num_spin_orbitals=1)
    #         self.assertEqual(simplified_op, targ)
    #
    #     with self.subTest("simplify complex"):
    #         fer_op = FermionicOp({"+_0 -_0": 1, "+_0 -_0 +_0 -_0": 1j}, num_spin_orbitals=1)
    #         simplified_op = fer_op.simplify()
    #         targ = FermionicOp({"+_0 -_0": 1 + 1j}, num_spin_orbitals=1)
    #         self.assertEqual(simplified_op, targ)
    #
    #     with self.subTest("simplify doesn't reorder"):
    #         fer_op = FermionicOp({"-_0 +_1": 1 + 0j}, num_spin_orbitals=2)
    #         simplified_op = fer_op.simplify()
    #         self.assertEqual(simplified_op, fer_op)
    #
    #     with self.subTest("simplify zero"):
    #         fer_op = self.op1 - self.op1
    #         simplified_op = fer_op.simplify()
    #         targ = FermionicOp.zero()
    #         self.assertEqual(simplified_op, targ)

    # def test_compose(self):
    #     """Test operator composition"""
    #     with self.subTest("single compose"):
    #         fer_op = FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2) @ FermionicOp(
    #             {"-_0": 1}, num_spin_orbitals=2
    #         )
    #         targ = FermionicOp({"+_0 -_1 -_0": 1}, num_spin_orbitals=2)
    #         self.assertEqual(fer_op, targ)
    #
    #     with self.subTest("multi compose"):
    #         fer_op = FermionicOp(
    #             {"+_0 +_1 -_1": 1, "-_0 +_0 -_1": 1}, num_spin_orbitals=2
    #         ) @ FermionicOp({"": 1, "-_0 +_1": 1}, num_spin_orbitals=2)
    #         fer_op = fer_op.simplify()
    #         targ = FermionicOp(
    #             {"+_0 +_1 -_1": 1, "-_0 +_0 -_1": 1, "+_0 -_0 +_1": 1, "-_0 -_1 +_1": -1},
    #             num_spin_orbitals=2,
    #         )
    #         self.assertEqual(fer_op, targ)

    # def setUp(self):
    #     super().setUp()
        # self.heisenberg_spin_array = np.array(
        #     [
        #         [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
        #         [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]],
        #         [[0, 0], [0, 0], [1, 1], [1, 0], [0, 1]],
        #     ],
        # )
        # self.heisenberg_coeffs = np.array([-1, -1, -1, -0.3, -0.3])
        # self.heisenberg = SpinOp(
        #     (self.heisenberg_spin_array, self.heisenberg_coeffs),
        #     spin=1,
        # )
        # self.zero_op = SpinOp(
        #     (np.array([[[0, 0]], [[0, 0]], [[0, 0]]]), np.array([0])),
        #     spin=1,
        # )
        # self.spin_1_matrix = {
        #     "I": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        #     "X": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2),
        #     "Y": np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2),
        #     "Z": np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
        # }

    # @staticmethod
    # def assertSpinEqual(first: SpinOp, second: SpinOp):
    #     """Fail if two SpinOps have different matrix representations."""
    #     np.testing.assert_array_almost_equal(first.to_matrix(), second.to_matrix())

    def test_init_label(self):
        # """Test __init__"""
        # print("label: ", label)
        # print("pp: ", pre_processing(label))
        #
        # spin = SpinOp(pre_processing(label), register_length=len(label) // 3)
        # expected_label = " ".join(lb for lb in label.split() if lb[0] != "I")
        # if not expected_label:
        #     expected_label = f"I_{len(label) // 3 - 1}"
        # self.assertListEqual(spin.to_list(), [(expected_label, 1)])
        # self.assertSpinEqual(eval(repr(spin)), spin)  # pylint: disable=eval-used
        # label = [("X_0 X_1", 3), ("Y_0 X_1", 3j), ("X_0 Y_1", -3j), ("Y_0 Y_1", 3)]
        label = {"X_0 X_1": 3, "Y_0 X_1": 3}
        label2 = {"X_0^2 X_1^2": 3, "Y_0^2 X_1^2": 3}
        label2 = {"X_0^2 X_1^2": 3, "Y_0 X_1": 1}
        print("label", label)
        spin = SpinOp(data=label2, spin = 2, num_orbitals = 2)
        # print(spin)
        # print("terms")
        # for t in spin.terms():
        #     print(t)
        spin_op = spin.simplify()
        print(spin_op)

    # def test_simplify(self):
    #     """Test simplify"""
    #     with self.subTest("simplify integer"):
    #         fer_op = FermionicOp({"+_0 -_0": 1, "+_0 -_0 +_0 -_0": 1}, num_spin_orbitals=1)
    #         simplified_op = fer_op.simplify()
    #         targ = FermionicOp({"+_0 -_0": 2}, num_spin_orbitals=1)
    #         self.assertEqual(simplified_op, targ)
    #
    #     with self.subTest("simplify complex"):
    #         fer_op = FermionicOp({"+_0 -_0": 1, "+_0 -_0 +_0 -_0": 1j}, num_spin_orbitals=1)
    #         simplified_op = fer_op.simplify()
    #         targ = FermionicOp({"+_0 -_0": 1 + 1j}, num_spin_orbitals=1)
    #         self.assertEqual(simplified_op, targ)
    #
    #     with self.subTest("simplify doesn't reorder"):
    #         fer_op = FermionicOp({"-_0 +_1": 1 + 0j}, num_spin_orbitals=2)
    #         simplified_op = fer_op.simplify()
    #         self.assertEqual(simplified_op, fer_op)
    #
    #     with self.subTest("simplify zero"):
    #         fer_op = self.op1 - self.op1
    #         simplified_op = fer_op.simplify()
    #         targ = FermionicOp.zero()
    #         self.assertEqual(simplified_op, targ)

    # @data(
    #     *product(
    #         (*sparse_labels(1), *sparse_labels(2), *sparse_labels(3)),
    #         (str2str, str2tuple, str2list),
    #     )
    # )
    # @unpack
    # def test_init_label(self, label, pre_processing):
        # """Test __init__"""
        # print("label: ", label)
        # print("pp: ", pre_processing(label))
        #
        # spin = SpinOp(pre_processing(label), register_length=len(label) // 3)
        # expected_label = " ".join(lb for lb in label.split() if lb[0] != "I")
        # if not expected_label:
        #     expected_label = f"I_{len(label) // 3 - 1}"
        # self.assertListEqual(spin.to_list(), [(expected_label, 1)])
        # self.assertSpinEqual(eval(repr(spin)), spin)  # pylint: disable=eval-used

    # @data(
    #     *product(
    #         (
    #             *zip(dense_labels(1), sparse_labels(1)),
    #             *zip(dense_labels(2), sparse_labels(2)),
    #             *zip(dense_labels(3), sparse_labels(3)),
    #         ),
    #         (str2str, str2tuple, str2list),
    #     )
    # )
    # @unpack
    # def test_init_dense_label(self, labels, pre_processing):
    #     """Test __init__ for dense label"""
    #     dense_label, sparse_label = labels
    #     actual = SpinOp(pre_processing(dense_label))
    #     desired = SpinOp([(sparse_label, 1)], register_length=len(dense_label))
    #     self.assertSpinEqual(actual, desired)
    #
    # def test_init_pm_label(self):
    #     """Test __init__ with plus and minus label"""
    #     with self.subTest("plus"):
    #         plus = SpinOp([("+_0", 2)], register_length=1)
    #         desired = SpinOp([("X_0", 2), ("Y_0", 2j)], register_length=1)
    #         self.assertSpinEqual(plus, desired)
    #
    #     with self.subTest("dense plus"):
    #         plus = SpinOp([("+", 2)])
    #         desired = SpinOp([("X_0", 2), ("Y_0", 2j)], register_length=1)
    #         self.assertSpinEqual(plus, desired)
    #
    #     with self.subTest("minus"):
    #         minus = SpinOp([("-_0", 2)], register_length=1)
    #         desired = SpinOp([("X_0", 2), ("Y_0", -2j)], register_length=1)
    #         self.assertSpinEqual(minus, desired)
    #
    #     with self.subTest("minus"):
    #         minus = SpinOp([("-", 2)])
    #         desired = SpinOp([("X_0", 2), ("Y_0", -2j)], register_length=1)
    #         self.assertSpinEqual(minus, desired)
    #
    #     with self.subTest("plus tensor minus"):
    #         plus_tensor_minus = SpinOp([("+_0 -_1", 3)], register_length=2)
    #         desired = SpinOp(
    #             [("X_0 X_1", 3), ("Y_0 X_1", 3j), ("X_0 Y_1", -3j), ("Y_0 Y_1", 3)],
    #             register_length=2,
    #         )
    #         self.assertSpinEqual(plus_tensor_minus, desired)
    #
    #     with self.subTest("dense plus tensor minus"):
    #         plus_tensor_minus = SpinOp([("+-", 3)])
    #         desired = SpinOp(
    #             [("X_0 X_1", 3), ("Y_0 X_1", 3j), ("X_0 Y_1", -3j), ("Y_0 Y_1", 3)],
    #             register_length=2,
    #         )
    #         self.assertSpinEqual(plus_tensor_minus, desired)
    #
    # def test_init_heisenberg(self):
    #     """Test __init__ for Heisenberg model."""
    #     actual = SpinOp(
    #         [
    #             ("XX", -1),
    #             ("YY", -1),
    #             ("ZZ", -1),
    #             ("ZI", -0.3),
    #             ("IZ", -0.3),
    #         ],
    #         spin=1,
    #     )
    #     self.assertSpinEqual(actual, self.heisenberg)
    #
    # def test_init_multiple_digits(self):
    #     """Test __init__ for sparse label with multiple digits"""
    #     actual = SpinOp([("X_10^20", 1 + 2j), ("X_12^34", 56)], Fraction(5, 2), register_length=13)
    #     desired = [("X_10^20", 1 + 2j), ("X_12^34", 56)]
    #     self.assertListEqual(actual.to_list(), desired)
    #
    # @data("IJX", "Z_0 X_0", "Z_0 +_0", "+_0 X_0")
    # def test_init_invalid_label(self, label):
    #     """Test __init__ for invalid label"""
    #     with self.assertRaises(ValueError):
    #         SpinOp(label)
    #
    # def test_init_raising_lowering_ops(self):
    #     """Test __init__ for +_i -_i pattern"""
    #     with self.subTest("one reg"):
    #         actual = SpinOp("+_0 -_0", spin=1, register_length=1)
    #         expected = SpinOp([("X_0^2", 1), ("Y_0^2", 1), ("Z_0", 1)], spin=1, register_length=1)
    #         self.assertSpinEqual(actual, expected)
    #     with self.subTest("two reg"):
    #         actual = SpinOp("+_1 -_1 +_0 -_0", spin=3 / 2, register_length=2)
    #         expected = SpinOp(
    #             [
    #                 ("X_0^2 X_1^2", 1),
    #                 ("X_0^2 Y_1^2", 1),
    #                 ("X_0^2 Z_1", 1),
    #                 ("Y_0^2 X_1^2", 1),
    #                 ("Y_0^2 Y_1^2", 1),
    #                 ("Y_0^2 Z_1", 1),
    #                 ("Z_0 X_1^2", 1),
    #                 ("Z_0 Y_1^2", 1),
    #                 ("Z_0 Z_1", 1),
    #             ],
    #             spin=3 / 2,
    #             register_length=2,
    #         )
    #         self.assertSpinEqual(actual, expected)
    #
    # def test_neg(self):
    #     """Test __neg__"""
    #     actual = -self.heisenberg
    #     desired = SpinOp((self.heisenberg_spin_array, -self.heisenberg_coeffs), spin=1)
    #     self.assertSpinEqual(actual, desired)
    #
    # def test_mul(self):
    #     """Test __mul__, and __rmul__"""
    #     actual = self.heisenberg * 2
    #     desired = SpinOp((self.heisenberg_spin_array, 2 * self.heisenberg_coeffs), spin=1)
    #     self.assertSpinEqual(actual, desired)
    #
    # def test_div(self):
    #     """Test __truediv__"""
    #     actual = self.heisenberg / 3
    #     desired = SpinOp((self.heisenberg_spin_array, self.heisenberg_coeffs / 3), spin=1)
    #     self.assertSpinEqual(actual, desired)
    #
    # def test_add(self):
    #     """Test __add__"""
    #     with self.subTest("sum of heisenberg"):
    #         actual = self.heisenberg + self.heisenberg
    #         desired = SpinOp((self.heisenberg_spin_array, 2 * self.heisenberg_coeffs), spin=1)
    #         self.assertSpinEqual(actual, desired)
    #
    #     with self.subTest("raising operator"):
    #         plus = SpinOp("+", 3 / 2)
    #         x = SpinOp("X", 3 / 2)
    #         y = SpinOp("Y", 3 / 2)
    #         self.assertSpinEqual(x + 1j * y, plus)
    #
    # def test_sub(self):
    #     """Test __sub__"""
    #     actual = self.heisenberg - self.heisenberg
    #     self.assertSpinEqual(actual, self.zero_op)
    #
    # def test_adjoint(self):
    #     """Test adjoint method"""
    #     with self.subTest("heisenberg adjoint"):
    #         actual = self.heisenberg.adjoint()
    #         desired = SpinOp(
    #             (self.heisenberg_spin_array, self.heisenberg_coeffs.conjugate().T),
    #             spin=1,
    #         )
    #         self.assertSpinEqual(actual, desired)
    #
    #     with self.subTest("imag heisenberg adjoint"):
    #         actual = ~((3 + 2j) * self.heisenberg)
    #         desired = SpinOp(
    #             (
    #                 self.heisenberg_spin_array,
    #                 ((3 + 2j) * self.heisenberg_coeffs).conjugate().T,
    #             ),
    #             spin=1,
    #         )
    #         self.assertSpinEqual(actual, desired)
    #
    #     # TODO: implement adjoint for same register operators.
    #     # with self.sub Test("adjoint same register op"):
    #     #     actual = SpinOp("X_0 Y_0 Z_0").adjoint()
    #
    #     #     print(actual.to_matrix())
    #     #     print(SpinOp("X_0 Y_0 Z_0").to_matrix().T.conjugate())
    #
    # def test_simplify(self):
    #     """Test simplify"""
    #     with self.subTest("trivial reduce"):
    #         actual = (self.heisenberg - self.heisenberg).simplify()
    #         self.assertListEqual(actual.to_list(), [("I_1", 0)])
    #
    #     with self.subTest("nontrivial reduce"):
    #         test_op = SpinOp(
    #             (
    #                 np.array([[[0, 1], [0, 1]], [[0, 0], [0, 0]], [[1, 0], [1, 0]]]),
    #                 np.array([1.5, 2.5]),
    #             ),
    #             spin=3 / 2,
    #         )
    #         actual = test_op.simplify()
    #         self.assertListEqual(actual.to_list(), [("Z_0 X_1", 4)])
    #
    #     with self.subTest("nontrivial reduce 2"):
    #         test_op = SpinOp(
    #             (
    #                 np.array(
    #                     [
    #                         [[0, 1], [0, 1], [1, 1]],
    #                         [[0, 0], [0, 0], [0, 0]],
    #                         [[1, 0], [1, 0], [0, 0]],
    #                     ]
    #                 ),
    #                 np.array([1.5, 2.5, 2]),
    #             ),
    #             spin=3 / 2,
    #         )
    #         actual = test_op.simplify()
    #         self.assertListEqual(actual.to_list(), [("Z_0 X_1", 4), ("X_0 X_1", 2)])
    #
    #     with self.subTest("nontrivial reduce 3"):
    #         test_op = SpinOp([("+_0 -_0", 1)], register_length=4)
    #         actual = test_op.simplify()
    #         self.assertListEqual(actual.to_list(), [("Z_0", 1), ("Y_0^2", 1), ("X_0^2", 1)])
    #
    # @data(*dense_labels(1))
    # def test_to_matrix_single_qutrit(self, label):
    #     """Test to_matrix for single qutrit op"""
    #     actual = SpinOp(label, 1).to_matrix()
    #     np.testing.assert_array_almost_equal(actual, self.spin_1_matrix[label])
    #
    # @data(*product(dense_labels(1), dense_labels(1)))
    # @unpack
    # def test_to_matrix_sum_single_qutrit(self, label1, label2):
    #     """Test to_matrix for sum qutrit op"""
    #     actual = (SpinOp(label1, 1) + SpinOp(label2, 1)).to_matrix()
    #     np.testing.assert_array_almost_equal(
    #         actual, self.spin_1_matrix[label1] + self.spin_1_matrix[label2]
    #     )
    #
    # @data(*dense_labels(2))
    # def test_to_matrix_two_qutrit(self, label):
    #     """Test to_matrix for two qutrit op"""
    #     actual = SpinOp(label, 1).to_matrix()
    #     desired = np.kron(self.spin_1_matrix[label[0]], self.spin_1_matrix[label[1]])
    #     np.testing.assert_array_almost_equal(actual, desired)
    #
    # @data(*dense_labels(1), *dense_labels(2), *dense_labels(3))
    # def test_consistency_with_pauli(self, label):
    #     """Test consistency with pauli"""
    #     actual = SpinOp(label).to_matrix()
    #     desired = Pauli(label).to_matrix() / (2 ** (len(label) - label.count("I")))
    #     np.testing.assert_array_almost_equal(actual, desired)
    #
    # def test_flatten_ladder_ops(self):
    #     """Test _flatten_ladder_ops"""
    #     actual = SpinOp._flatten_ladder_ops([("+-", 2j)])
    #     self.assertSetEqual(
    #         frozenset(actual),
    #         frozenset([("XX", 2j), ("XY", 2), ("YX", -2), ("YY", 2j)]),
    #     )
    #
    # def test_hermiticity(self):
    #     """test is_hermitian"""
    #     # deliberately define test operator with X and Y which creates duplicate terms in .to_list()
    #     # in case .adjoint() simplifies terms
    #     with self.subTest("operator hermitian"):
    #         test_op = SpinOp("+ZXY") + SpinOp("-ZXY")
    #         self.assertTrue(test_op.is_hermitian())
    #
    #     with self.subTest("operator not hermitian"):
    #         test_op = SpinOp("+ZXY") - SpinOp("-ZXY")
    #         self.assertFalse(test_op.is_hermitian())
    #
    #     with self.subTest("test passing atol"):
    #         test_op = SpinOp("+ZXY") + (1 + 1e-7) * SpinOp("-ZXY")
    #         self.assertFalse(test_op.is_hermitian())
    #         self.assertFalse(test_op.is_hermitian(atol=1e-8))
    #         self.assertTrue(test_op.is_hermitian(atol=1e-6))
    #
    # def test_equiv(self):
    #     """test equiv"""
    #     op1 = SpinOp("+ZXY") + SpinOp("-XXX")
    #     op2 = SpinOp("+ZXY")
    #     op3 = SpinOp("+ZXY") + (1 + 1e-7) * SpinOp("-XXX")
    #     self.assertFalse(op1.equiv(op2))
    #     self.assertFalse(op1.equiv(op3))
    #     self.assertTrue(op1.equiv(op3, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
