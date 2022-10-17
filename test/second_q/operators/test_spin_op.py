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

if __name__ == "__main__":
    unittest.main()
