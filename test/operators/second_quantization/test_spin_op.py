# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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
from itertools import product
from test import QiskitNatureTestCase
from typing import Callable, Optional

import numpy as np
from ddt import data, ddt, unpack
from qiskit.quantum_info import Pauli

from qiskit_nature.operators import SpinOp


def spin_labels(length):
    """Generate list of spin labels with given length."""
    return ["".join(label) for label in product(["I", "X", "Y", "Z"], repeat=length)]


@ddt
class TestSpinOp(QiskitNatureTestCase):
    """SpinOp tests."""

    def setUp(self):
        super().setUp()
        self.heisenberg_spin_array = np.array(
            [
                [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [1, 1], [1, 0], [0, 1]],
            ],
        )
        self.heisenberg_coeffs = np.array([-1, -1, -1, -0.3, -0.3])
        self.heisenberg = SpinOp(
            (self.heisenberg_spin_array, self.heisenberg_coeffs),
            spin=1,
        )
        self.zero_op = SpinOp(
            (np.array([[[0, 0]], [[0, 0]], [[0, 0]]]), np.array([0])),
            spin=1,
        )
        self.spin_1_matrix = {
            "I": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "X": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2),
            "Y": np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2),
            "Z": np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
        }

    @staticmethod
    def assertSpinEqual(first: SpinOp, second: SpinOp):
        """Fail if two SpinOps have different matrix representations."""
        np.testing.assert_array_almost_equal(first.to_matrix(), second.to_matrix())

    @data(*spin_labels(1))
    def test_init_label(self, label):
        """Test __init__"""
        spin = SpinOp(f"{label}_0", register_length=1)
        self.assertListEqual(spin.to_list(), [(f"{label}_0", 1)])

    @data(*spin_labels(2))
    def test_init_len2_label(self, label):
        """Test __init__"""
        spin = SpinOp(f"{label[1]}_0 {label[0]}_1", register_length=2)
        self.assertListEqual(spin.to_list(), [(f"{label[1]}_0 {label[0]}_1", 1)])

    def test_init_pm_label(self):
        """Test __init__ with plus and minus label"""
        with self.subTest("plus"):
            plus = SpinOp([("+_0", 2)], register_length=1)
            desired = SpinOp([("X_0", 2), ("Y_0", 2j)], register_length=1)
            self.assertSpinEqual(plus, desired)

        with self.subTest("dense plus"):
            plus = SpinOp([("+", 2)])
            desired = SpinOp([("X_0", 2), ("Y_0", 2j)], register_length=1)
            self.assertSpinEqual(plus, desired)

        with self.subTest("minus"):
            minus = SpinOp([("-_0", 2)], register_length=1)
            desired = SpinOp([("X_0", 2), ("Y_0", -2j)], register_length=1)
            self.assertSpinEqual(minus, desired)

        with self.subTest("minus"):
            minus = SpinOp([("-", 2)])
            desired = SpinOp([("X_0", 2), ("Y_0", -2j)], register_length=1)
            self.assertSpinEqual(minus, desired)

        with self.subTest("plus tensor minus"):
            plus_tensor_minus = SpinOp([("+_0 -_1", 3)], register_length=2)
            desired = SpinOp(
                [("X_0 X_1", 3), ("Y_0 X_1", 3j), ("X_0 Y_1", -3j), ("Y_0 Y_1", 3)],
                register_length=2,
            )
            self.assertSpinEqual(plus_tensor_minus, desired)

        with self.subTest("dense plus tensor minus"):
            plus_tensor_minus = SpinOp([("+-", 3)])
            desired = SpinOp(
                [("X_0 X_1", 3), ("Y_0 X_1", 3j), ("X_0 Y_1", -3j), ("Y_0 Y_1", 3)],
                register_length=2,
            )
            self.assertSpinEqual(plus_tensor_minus, desired)

    def test_init_heisenberg(self):
        """Test __init__ for Heisenberg model."""
        actual = SpinOp(
            [
                ("XX", -1),
                ("YY", -1),
                ("ZZ", -1),
                ("ZI", -0.3),
                ("IZ", -0.3),
            ],
            spin=1,
        )
        self.assertSpinEqual(actual, self.heisenberg)

    @data(*spin_labels(1), *spin_labels(2))
    def test_init_dense_label(self, label):
        """Test __init__ for dense label"""
        if len(label) == 1:
            actual = SpinOp([(f"{label}", 1 + 1j)])
            desired = SpinOp([(f"{label}_0", 1 + 1j)], register_length=1)
        elif len(label) == 2:
            actual = SpinOp([(f"{label}", 1)])
            desired = SpinOp([(f"{label[0]}_0 {label[1]}_1", 1)], register_length=2)
        self.assertSpinEqual(actual, desired)

    def test_init_multiple_digits(self):
        """Test __init__ for sparse label with multiple digits"""
        actual = SpinOp([("X_10^20", 1 + 2j), ("X_12^34", 56)], Fraction(5, 2), register_length=13)
        desired = [
            ("I_0 I_1 I_2 I_3 I_4 I_5 I_6 I_7 I_8 I_9 X_10^20 I_11 I_12", 1 + 2j),
            ("I_0 I_1 I_2 I_3 I_4 I_5 I_6 I_7 I_8 I_9 I_10 I_11 X_12^34", 56),
        ]
        self.assertListEqual(actual.to_list(), desired)

    @data("IJX", "Z_0 X_0", "Z_0 +_0", "+_0 X_0")
    def test_init_invalid_label(self, label):
        """Test __init__ for invalid label"""
        with self.assertRaises(ValueError):
            SpinOp(label)

    def test_init_raising_lowering_ops(self):
        """Test __init__ for +_i -_i pattern"""
        with self.subTest("one reg"):
            actual = SpinOp("+_0 -_0", spin=1, register_length=1)
            expected = SpinOp([("X_0^2", 1), ("Y_0^2", 1), ("Z_0", 1)], spin=1, register_length=1)
            self.assertSpinEqual(actual, expected)
        with self.subTest("two reg"):
            actual = SpinOp("+_1 -_1 +_0 -_0", spin=3 / 2, register_length=2)
            expected = SpinOp(
                [
                    ("X_0^2 X_1^2", 1),
                    ("X_0^2 Y_1^2", 1),
                    ("X_0^2 Z_1", 1),
                    ("Y_0^2 X_1^2", 1),
                    ("Y_0^2 Y_1^2", 1),
                    ("Y_0^2 Z_1", 1),
                    ("Z_0 X_1^2", 1),
                    ("Z_0 Y_1^2", 1),
                    ("Z_0 Z_1", 1),
                ],
                spin=3 / 2,
                register_length=2,
            )
            self.assertSpinEqual(actual, expected)

    def test_neg(self):
        """Test __neg__"""
        actual = -self.heisenberg
        desired = SpinOp((self.heisenberg_spin_array, -self.heisenberg_coeffs), spin=1)
        self.assertSpinEqual(actual, desired)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        actual = self.heisenberg * 2
        desired = SpinOp((self.heisenberg_spin_array, 2 * self.heisenberg_coeffs), spin=1)
        self.assertSpinEqual(actual, desired)

    def test_div(self):
        """Test __truediv__"""
        actual = self.heisenberg / 3
        desired = SpinOp((self.heisenberg_spin_array, self.heisenberg_coeffs / 3), spin=1)
        self.assertSpinEqual(actual, desired)

    def test_add(self):
        """Test __add__"""
        with self.subTest("sum of heisenberg"):
            actual = self.heisenberg + self.heisenberg
            desired = SpinOp((self.heisenberg_spin_array, 2 * self.heisenberg_coeffs), spin=1)
            self.assertSpinEqual(actual, desired)

        with self.subTest("raising operator"):
            plus = SpinOp("+", 3 / 2)
            x = SpinOp("X", 3 / 2)
            y = SpinOp("Y", 3 / 2)
            self.assertSpinEqual(x + 1j * y, plus)

    def test_sub(self):
        """Test __sub__"""
        actual = self.heisenberg - self.heisenberg
        self.assertSpinEqual(actual, self.zero_op)

    def test_adjoint(self):
        """Test adjoint method and dagger property"""
        with self.subTest("heisenberg adjoint"):
            actual = self.heisenberg.adjoint()
            desired = SpinOp(
                (self.heisenberg_spin_array, self.heisenberg_coeffs.conjugate().T), spin=1
            )
            self.assertSpinEqual(actual, desired)

        with self.subTest("imag heisenberg adjoint"):
            actual = ~((3 + 2j) * self.heisenberg)
            desired = SpinOp(
                (self.heisenberg_spin_array, ((3 + 2j) * self.heisenberg_coeffs).conjugate().T),
                spin=1,
            )
            self.assertSpinEqual(actual, desired)

        # TODO: implement adjoint for same register operators.
        # with self.sub Test("adjoint same register op"):
        #     actual = SpinOp("X_0 Y_0 Z_0").dagger

        #     print(actual.to_matrix())
        #     print(SpinOp("X_0 Y_0 Z_0").to_matrix().T.conjugate())

    def test_reduce(self):
        """Test reduce"""
        with self.subTest("trivial reduce"):
            actual = (self.heisenberg - self.heisenberg).reduce()
            self.assertListEqual(actual.to_list(), [("I_0 I_1", 0)])

        with self.subTest("nontrivial reduce"):
            test_op = SpinOp(
                (
                    np.array([[[0, 1], [0, 1]], [[0, 0], [0, 0]], [[1, 0], [1, 0]]]),
                    np.array([1.5, 2.5]),
                ),
                spin=3 / 2,
            )
            actual = test_op.reduce()
            self.assertListEqual(actual.to_list(), [("Z_0 X_1", 4)])

        with self.subTest("nontrivial reduce 2"):
            test_op = SpinOp(
                (
                    np.array(
                        [
                            [[0, 1], [0, 1], [1, 1]],
                            [[0, 0], [0, 0], [0, 0]],
                            [[1, 0], [1, 0], [0, 0]],
                        ]
                    ),
                    np.array([1.5, 2.5, 2]),
                ),
                spin=3 / 2,
            )
            actual = test_op.reduce()
            self.assertListEqual(actual.to_list(), [("Z_0 X_1", 4), ("X_0 X_1", 2)])

    @data(*spin_labels(1))
    def test_to_matrix_single_qutrit(self, label):
        """Test to_matrix for single qutrit op"""
        actual = SpinOp(label, 1).to_matrix()
        np.testing.assert_array_almost_equal(actual, self.spin_1_matrix[label])

    @data(*product(spin_labels(1), spin_labels(1)))
    @unpack
    def test_to_matrix_sum_single_qutrit(self, label1, label2):
        """Test to_matrix for sum qutrit op"""
        actual = (SpinOp(label1, 1) + SpinOp(label2, 1)).to_matrix()
        np.testing.assert_array_almost_equal(
            actual, self.spin_1_matrix[label1] + self.spin_1_matrix[label2]
        )

    @data(*spin_labels(2))
    def test_to_matrix_two_qutrit(self, label):
        """Test to_matrix for two qutrit op"""
        actual = SpinOp(label, 1).to_matrix()
        desired = np.kron(self.spin_1_matrix[label[0]], self.spin_1_matrix[label[1]])
        np.testing.assert_array_almost_equal(actual, desired)

    @data(*spin_labels(1), *spin_labels(2), *spin_labels(3))
    def test_consistency_with_pauli(self, label):
        """Test consistency with pauli"""
        actual = SpinOp(label).to_matrix()
        desired = Pauli(label).to_matrix() / (2 ** (len(label) - label.count("I")))
        np.testing.assert_array_almost_equal(actual, desired)

    def test_flatten_ladder_ops(self):
        """Test _flatten_ladder_ops"""
        actual = SpinOp._flatten_ladder_ops([("+-", 2j)])
        self.assertSetEqual(
            frozenset(actual),
            frozenset([("XX", 2j), ("XY", 2), ("YX", -2), ("YY", 2j)]),
        )


if __name__ == "__main__":
    unittest.main()
