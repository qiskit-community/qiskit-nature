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

import numpy as np
from ddt import data, ddt

from qiskit_nature.operators import SpinOp


def spin_labels(length):
    """Generate list of fermion labels with given length."""
    return ["".join(label) for label in product(["I", "X", "Y", "Z"], repeat=length)]


@ddt
class TestSpinOp(QiskitNatureTestCase):
    """FermionicOp tests."""

    def setUp(self):
        super().setUp()
        heisenberg_spin_array = np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        heisenberg_coeffs = np.array([-1, -1, -1, -0.3, -0.3])
        self.heisenberg = SpinOp(
            (heisenberg_spin_array, heisenberg_coeffs),
            spin=1,
        )
        self.heisenberg_mat = self.heisenberg.to_matrix()

    @data(*spin_labels(1))
    def test_init_label(self, label):
        """Test __init__"""
        spin = SpinOp([(f"{label}_0", 1)])
        self.assertListEqual(spin.to_list(), [(f"{label}_0", 1)])

    def test_neg(self):
        """Test __neg__"""
        actual = -self.heisenberg
        desired = -self.heisenberg_mat
        np.testing.assert_array_almost_equal(actual.to_matrix(), desired)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        actual = self.heisenberg * 2
        np.testing.assert_array_almost_equal(actual.to_matrix(), self.heisenberg_mat * 2)

    def test_div(self):
        """Test __truediv__"""
        actual = self.heisenberg / 3
        np.testing.assert_array_almost_equal(actual.to_matrix(), self.heisenberg_mat / 3)

    def test_add(self):
        """Test __add__"""
        actual = self.heisenberg + self.heisenberg
        np.testing.assert_array_almost_equal(actual.to_matrix(), self.heisenberg_mat * 2)

    def test_sub(self):
        """Test __sub__"""
        actual = self.heisenberg - self.heisenberg
        np.testing.assert_array_almost_equal(actual.to_matrix(), np.zeros((9, 9)))

    def test_adjoint(self):
        """Test adjoint method and dagger property"""
        actual = ~self.heisenberg
        np.testing.assert_array_almost_equal(actual.to_matrix(), self.heisenberg_mat.conj().T)

    def test_reduce(self):
        """Test reduce"""
        actual = (self.heisenberg - self.heisenberg).reduce()
        self.assertListEqual(actual.to_list(), [("I_1", 0)])


if __name__ == "__main__":
    unittest.main()
