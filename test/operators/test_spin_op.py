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
from itertools import product
from test import QiskitNatureTestCase
from typing import Callable, Optional

import numpy as np
from ddt import data, ddt
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

    @staticmethod
    def assertSpinEqual(first: SpinOp, second: SpinOp):
        """Fail if two SpinOps have different matrix representations."""
        np.testing.assert_array_almost_equal(first.to_matrix(), second.to_matrix())

    @data(*spin_labels(1))
    def test_init_label(self, label):
        """Test __init__"""
        spin = SpinOp(f"{label}_0")
        self.assertListEqual(spin.to_list(), [(f"{label}_0", 1)])

    @data(*spin_labels(2))
    def test_init_len2_label(self, label):
        """Test __init__"""
        spin = SpinOp(f"{label[1]}_1 {label[0]}_0")
        self.assertListEqual(spin.to_list(), [(f"{label[1]}_1 {label[0]}_0", 1)])

    def test_init_pm_label(self):
        """Test __init__ with plus and minus label"""
        with self.subTest("plus"):
            plus = SpinOp([("+_0", 2)])
            desired = SpinOp([("X_0", 2), ("Y_0", 2j)])
            self.assertSpinEqual(plus, desired)

        with self.subTest("dense plus"):
            plus = SpinOp([("+", 2)])
            desired = SpinOp([("X_0", 2), ("Y_0", 2j)])
            self.assertSpinEqual(plus, desired)

        with self.subTest("minus"):
            minus = SpinOp([("-_0", 2)])
            desired = SpinOp([("X_0", 2), ("Y_0", -2j)])
            self.assertSpinEqual(minus, desired)

        with self.subTest("minus"):
            minus = SpinOp([("-", 2)])
            desired = SpinOp([("X_0", 2), ("Y_0", -2j)])
            self.assertSpinEqual(minus, desired)

        with self.subTest("plus tensor minus"):
            plus_tensor_minus = SpinOp([("+_1 -_0", 3)])
            desired = SpinOp([("X_1 X_0", 3), ("X_1 Y_0", -3j), ("Y_1 X_0", 3j), ("Y_1 Y_0", 3)])
            self.assertSpinEqual(plus_tensor_minus, desired)

        with self.subTest("dense plus tensor minus"):
            plus_tensor_minus = SpinOp([("+-", 3)])
            desired = SpinOp([("X_1 X_0", 3), ("X_1 Y_0", -3j), ("Y_1 X_0", 3j), ("Y_1 Y_0", 3)])
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
            desired = SpinOp([(f"{label}_0", 1 + 1j)])
        elif len(label) == 2:
            actual = SpinOp([(f"{label}", 1)])
            desired = SpinOp([(f"{label[0]}_1 {label[1]}_0", 1)])
        self.assertSpinEqual(actual, desired)

    @data("IJX", "Z_0 X_0", "Z_0 +_0", "+_0 X_0")
    def test_init_invalid_label(self, label):
        """Test __init__ for invalid label"""
        with self.assertRaises(ValueError):
            SpinOp(label)

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

    def test_reduce(self):
        """Test reduce"""
        with self.subTest("trivial reduce"):
            actual = (self.heisenberg - self.heisenberg).reduce()
            self.assertListEqual(actual.to_list(), [("I_1 I_0", 0)])

        with self.subTest("nontrivial reduce"):
            test_op = SpinOp(
                (
                    np.array([[[0, 1], [0, 1]], [[0, 0], [0, 0]], [[1, 0], [1, 0]]]),
                    np.array([1.5, 2.5]),
                ),
                spin=3 / 2,
            )
            actual = test_op.reduce()
            self.assertListEqual(actual.to_list(), [("Z_1 X_0", 4)])

    def test_consistency_with_pauli(self):
        """Test consistency with pauli"""
        actual = SpinOp("XYZ").to_matrix()
        desired = Pauli("XYZ").to_matrix() / 8
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
