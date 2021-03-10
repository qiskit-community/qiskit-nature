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

"""Test for FiniteBosonicOp"""

import unittest
from itertools import product
from test import QiskitNatureTestCase
from typing import Callable, Optional

import numpy as np
from ddt import data, ddt, unpack
from qiskit.quantum_info import Pauli

from qiskit_nature.operators import FiniteBosonicOp


def spin_labels(length):
    """Generate list of spin labels with given length."""
    return ["".join(label) for label in product(["I", "X", "Y", "Z"], repeat=length)]


@ddt
class TestFiniteBosonicOp(QiskitNatureTestCase):
    """FiniteBosonicOp tests."""

    def setUp(self):
        super().setUp()
        self.heisenberg_array = np.array(
            [
                [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [1, 1], [1, 0], [0, 1]],
            ],
        )
        self.heisenberg_coeffs = np.array([-1, -1, -1, -0.3, -0.3])
        self.heisenberg = FiniteBosonicOp(
            (self.heisenberg_array, self.heisenberg_coeffs),
            truncation_level=3,
        )
        self.zero_op = FiniteBosonicOp(
            (np.array([[[0, 0]], [[0, 0]], [[0, 0]]]), np.array([0])),
            truncation_level=3,
        )
        self.spin_1_matrix = {
            "I": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "X": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2),
            "Y": np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2),
            "Z": np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
        }

    @staticmethod
    def assertBosonEqual(first: FiniteBosonicOp, second: FiniteBosonicOp):
        """Fail if two FiniteBosonicOps have different matrix representations."""
        np.testing.assert_array_almost_equal(first.to_matrix(), second.to_matrix())

    @data(*spin_labels(1))
    def test_init_label(self, label):
        """Test __init__"""
        boson = FiniteBosonicOp(f"{label}_0")
        self.assertListEqual(boson.to_list(), [(f"{label}_0", 1)])

    @data(*spin_labels(2))
    def test_init_len2_label(self, label):
        """Test __init__"""
        boson = FiniteBosonicOp(f"{label[1]}_1 {label[0]}_0")
        self.assertListEqual(boson.to_list(), [(f"{label[1]}_1 {label[0]}_0", 1)])

    def test_init_pm_label(self):
        """Test __init__ with plus and minus label"""
        with self.subTest("plus"):
            plus = FiniteBosonicOp([("+_0", 2)])
            desired = FiniteBosonicOp([("X_0", 2), ("Y_0", 2j)])
            self.assertBosonEqual(plus, desired)

        with self.subTest("dense plus"):
            plus = FiniteBosonicOp([("+", 2)])
            desired = FiniteBosonicOp([("X_0", 2), ("Y_0", 2j)])
            self.assertBosonEqual(plus, desired)

        with self.subTest("minus"):
            minus = FiniteBosonicOp([("-_0", 2)])
            desired = FiniteBosonicOp([("X_0", 2), ("Y_0", -2j)])
            self.assertBosonEqual(minus, desired)

        with self.subTest("minus"):
            minus = FiniteBosonicOp([("-", 2)])
            desired = FiniteBosonicOp([("X_0", 2), ("Y_0", -2j)])
            self.assertBosonEqual(minus, desired)

        with self.subTest("plus tensor minus"):
            plus_tensor_minus = FiniteBosonicOp([("+_1 -_0", 3)])
            desired = FiniteBosonicOp(
                [("X_1 X_0", 3), ("X_1 Y_0", -3j), ("Y_1 X_0", 3j), ("Y_1 Y_0", 3)]
            )
            self.assertBosonEqual(plus_tensor_minus, desired)

        with self.subTest("dense plus tensor minus"):
            plus_tensor_minus = FiniteBosonicOp([("+-", 3)])
            desired = FiniteBosonicOp(
                [("X_1 X_0", 3), ("X_1 Y_0", -3j), ("Y_1 X_0", 3j), ("Y_1 Y_0", 3)]
            )
            self.assertBosonEqual(plus_tensor_minus, desired)

    def test_init_heisenberg(self):
        """Test __init__ for Heisenberg model."""
        actual = FiniteBosonicOp(
            [
                ("XX", -1),
                ("YY", -1),
                ("ZZ", -1),
                ("ZI", -0.3),
                ("IZ", -0.3),
            ],
            truncation_level=3,
        )
        self.assertBosonEqual(actual, self.heisenberg)

    @data(*spin_labels(1), *spin_labels(2))
    def test_init_dense_label(self, label):
        """Test __init__ for dense label"""
        if len(label) == 1:
            actual = FiniteBosonicOp([(f"{label}", 1 + 1j)])
            desired = FiniteBosonicOp([(f"{label}_0", 1 + 1j)])
        elif len(label) == 2:
            actual = FiniteBosonicOp([(f"{label}", 1)])
            desired = FiniteBosonicOp([(f"{label[0]}_1 {label[1]}_0", 1)])
        self.assertBosonEqual(actual, desired)

    @data("IJX", "Z_0 X_0", "Z_0 +_0", "+_0 X_0")
    def test_init_invalid_label(self, label):
        """Test __init__ for invalid label"""
        with self.assertRaises(ValueError):
            FiniteBosonicOp(label)

    def test_neg(self):
        """Test __neg__"""
        actual = -self.heisenberg
        desired = FiniteBosonicOp(
            (self.heisenberg_array, -self.heisenberg_coeffs), truncation_level=3
        )
        self.assertBosonEqual(actual, desired)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        actual = self.heisenberg * 2
        desired = FiniteBosonicOp(
            (self.heisenberg_array, 2 * self.heisenberg_coeffs), truncation_level=3
        )
        self.assertBosonEqual(actual, desired)

    def test_div(self):
        """Test __truediv__"""
        actual = self.heisenberg / 3
        desired = FiniteBosonicOp(
            (self.heisenberg_array, self.heisenberg_coeffs / 3), truncation_level=3
        )
        self.assertBosonEqual(actual, desired)

    def test_add(self):
        """Test __add__"""
        with self.subTest("sum of heisenberg"):
            actual = self.heisenberg + self.heisenberg
            desired = FiniteBosonicOp(
                (self.heisenberg_array, 2 * self.heisenberg_coeffs), truncation_level=3
            )
            self.assertBosonEqual(actual, desired)

        with self.subTest("raising operator"):
            plus = FiniteBosonicOp("+", 4)
            x = FiniteBosonicOp("X", 4)
            y = FiniteBosonicOp("Y", 4)
            self.assertBosonEqual(x + 1j * y, plus)

    def test_sub(self):
        """Test __sub__"""
        actual = self.heisenberg - self.heisenberg
        self.assertBosonEqual(actual, self.zero_op)

    def test_adjoint(self):
        """Test adjoint method and dagger property"""
        with self.subTest("heisenberg adjoint"):
            actual = self.heisenberg.adjoint()
            desired = FiniteBosonicOp(
                (self.heisenberg_array, self.heisenberg_coeffs.conjugate().T), truncation_level=3
            )
            self.assertBosonEqual(actual, desired)

        with self.subTest("imag heisenberg adjoint"):
            actual = ~((3 + 2j) * self.heisenberg)
            desired = FiniteBosonicOp(
                (self.heisenberg_array, ((3 + 2j) * self.heisenberg_coeffs).conjugate().T),
                truncation_level=3,
            )
            self.assertBosonEqual(actual, desired)

        # TODO: implement adjoint for same register operators.
        # with self.sub Test("adjoint same register op"):
        #     actual = FiniteBosonicOp("X_0 Y_0 Z_0").dagger

        #     print(actual.to_matrix())
        #     print(FiniteBosonicOp("X_0 Y_0 Z_0").to_matrix().T.conjugate())

    def test_reduce(self):
        """Test reduce"""
        with self.subTest("trivial reduce"):
            actual = (self.heisenberg - self.heisenberg).reduce()
            self.assertListEqual(actual.to_list(), [("I_1 I_0", 0)])

        with self.subTest("nontrivial reduce"):
            test_op = FiniteBosonicOp(
                (
                    np.array([[[0, 1], [0, 1]], [[0, 0], [0, 0]], [[1, 0], [1, 0]]]),
                    np.array([1.5, 2.5]),
                ),
                truncation_level=4,
            )
            actual = test_op.reduce()
            self.assertListEqual(actual.to_list(), [("Z_1 X_0", 4)])

        with self.subTest("nontrivial reduce 2"):
            test_op = FiniteBosonicOp(
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
                truncation_level=4,
            )
            actual = test_op.reduce()
            self.assertListEqual(actual.to_list(), [("Z_1 X_0", 4), ("X_1 X_0", 2)])

    @data(*spin_labels(1))
    def test_to_matrix_single_qutrit(self, label):
        """Test to_matrix for single qutrit op"""
        actual = FiniteBosonicOp(label, 3).to_matrix()
        np.testing.assert_array_almost_equal(actual, self.spin_1_matrix[label])

    @data(*product(spin_labels(1), spin_labels(1)))
    @unpack
    def test_to_matrix_sum_single_qutrit(self, label1, label2):
        """Test to_matrix for sum qutrit op"""
        actual = (FiniteBosonicOp(label1, 3) + FiniteBosonicOp(label2, 3)).to_matrix()
        np.testing.assert_array_almost_equal(
            actual, self.spin_1_matrix[label1] + self.spin_1_matrix[label2]
        )

    @data(*spin_labels(2))
    def test_to_matrix_two_qutrit(self, label):
        """Test to_matrix for two qutrit op"""
        actual = FiniteBosonicOp(label, 3).to_matrix()
        expected = np.kron(self.spin_1_matrix[label[0]], self.spin_1_matrix[label[1]])
        np.testing.assert_array_almost_equal(actual, expected)

    @data(*spin_labels(1), *spin_labels(2), *spin_labels(3))
    def test_consistency_with_pauli(self, label):
        """Test consistency with pauli"""
        actual = FiniteBosonicOp(label).to_matrix()
        expected = Pauli(label).to_matrix() / (2 ** (len(label) - label.count("I")))
        np.testing.assert_array_almost_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
