# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for MixedOp"""

import unittest
import warnings
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt, unpack
from qiskit.circuit import Parameter
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs

from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.operators import FermionicOp, MixedOp
import qiskit_nature.optionals as _optionals

from qiskit_nature import settings


@ddt
class TestFermionicOp(QiskitNatureTestCase):
    """FermionicOp tests."""

    op1 = FermionicOp({"+_0 -_0": 1})
    op2 = FermionicOp({"-_0 +_0": 2})
    mop1_h1 = MixedOp({("h1",): [(2.0, op1)]})
    mop2_h1 = MixedOp({("h1",): [(3.0, op2)]})
    mop2_h2 = MixedOp({("h2",): [(3.0, op2)]})
    sumop2 = MixedOp({("h1",): [(1, FermionicOp({"+_0 -_0": 2, "-_0 +_0": 3}))]})

    def test_neg(self):
        """Test __neg__"""
        minus_mop1 = -self.mop1_h1
        targ_v1 = MixedOp({("h1",): [(-2.0, self.op1)]})
        self.assertEqual(minus_mop1, targ_v1)

        # Requires the simplify()
        # targ_v2 = MixedOp({("h1",): [(2.0, -self.op1)]})
        # self.assertEqual(minus_mop1, targ_v2)

        # fer_op = -self.op4
        # targ = FermionicOp({"+_0 -_0": -self.a})
        # self.assertEqual(fer_op, targ)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        with self.subTest("rightmul"):
            minus_mop1 = self.mop1_h1 * 2.0
            targ_v1 = MixedOp({("h1",): [(4.0, self.op1)]})
            self.assertEqual(minus_mop1, targ_v1)

            # targ_v2 = MixedOp({("h1",): [(2.0, 2.0 * self.op1)]})
            # self.assertEqual(minus_mop1, targ_v2)

            # Requires the simplify()
            # fer_op = self.op1 * self.a
            # targ = FermionicOp({"+_0 -_0": self.a})
            # self.assertEqual(fer_op, targ)

        with self.subTest("leftmul"):
            minus_mop1 = (2.0 + 1.0j) * self.mop1_h1
            targ_v1 = MixedOp({("h1",): [((4.0 + 2.0j), self.op1)]})
            self.assertEqual(minus_mop1, targ_v1)

            # Requires the simplify()
            # targ_v2 = MixedOp({("h1",): [(2.0, (2.0 + 1.0j) * self.op1)]})
            # self.assertEqual(minus_mop1, targ_v2)

    # def test_div(self):
    #     """Test __truediv__"""
    #     fer_op = self.op1 / 2
    #     targ = FermionicOp({"+_0 -_0": 0.5}, num_spin_orbitals=1)
    #     self.assertEqual(fer_op, targ)

    #     fer_op = self.op1 / self.a
    #     targ = FermionicOp({"+_0 -_0": 1 / self.a})
    #     self.assertEqual(fer_op, targ)

    def test_add(self):
        """Test __add__"""
        with self.subTest("same hilbert space"):
            sum_mop = self.mop1_h1 + self.mop2_h1
            targ = MixedOp({("h1",): [(2, self.op1), (3, self.op2)]})
            self.assertEqual(sum_mop, targ)

            # Requires the simplify()
            # targ = self.sumop2
            # self.assertEqual(sum_mop, targ)

        with self.subTest("different hilbert space"):
            sum_mop = self.mop1_h1 + self.mop2_h2
            targ = MixedOp({("h1",): [(2.0, self.op1)], ("h2",): [(3.0, self.op2)]})
            self.assertEqual(sum_mop, targ)

        # with self.subTest("sum"):
        #     fer_op = sum(FermionicOp({label: 1}) for label in ["+_0", "-_1", "+_2 -_2"])
        #     targ = FermionicOp({"+_0": 1, "-_1": 1, "+_2 -_2": 1})
        #     self.assertEqual(fer_op, targ)

    def test_sub(self):
        """Test __sub__"""
        with self.subTest("same hilbert space"):
            sum_mop = self.mop1_h1 - self.mop2_h1
            targ = MixedOp({("h1",): [(2, self.op1), (-3, self.op2)]})
            self.assertEqual(sum_mop, targ)

            # Requires the simplify()
            # targ = self.sumop2
            # self.assertEqual(sum_mop, targ)

        with self.subTest("different hilbert space"):
            sum_mop = self.mop1_h1 - self.mop2_h2
            targ = MixedOp({("h1",): [(2.0, self.op1)], ("h2",): [(-3.0, self.op2)]})
            self.assertEqual(sum_mop, targ)

    def test_compose(self):
        """Test operator composition"""
        with self.subTest("same hilbert spaces"):
            composed_op = self.mop1_h1.compose(self.mop2_h1)
            targ = MixedOp({("h1", "h1"): [(6.0, self.op1, self.op2)]})
            self.assertEqual(composed_op, targ)

        with self.subTest("different hilbert spaces"):
            composed_op = self.mop1_h1.compose(self.mop2_h2)
            targ = MixedOp({("h1", "h2"): [(6.0, self.op1, self.op2)]})
            self.assertEqual(composed_op, targ)

    def test_adjoint(self):
        """Test adjoint method"""
        test_adjoint = MixedOp(
            {
                ("h1",): [(2.0 + 1j, self.op1)],
                (
                    "h1",
                    "h2",
                ): [(3.0 +2j, self.op1, self.op2)],
                "h2": [(4.0 -5j, self.op2)],
            }
        ).adjoint()

        targ_adjoint = MixedOp(
            {
                ("h1",): [(2.0 - 1j, self.op1.adjoint())],
                (
                    "h1",
                    "h2",
                ): [(3.0 -2j, self.op1.adjoint(), self.op2.adjoint())],
                "h2": [(4.0 +5j, self.op2.adjoint())],
            }
        )
        self.assertEqual(test_adjoint, targ_adjoint)

    def test_conjugate(self):
        """Test conjugate method"""
        test_conjugate = MixedOp(
            {
                ("h1",): [(2.0 + 1j, self.op1)],
                (
                    "h1",
                    "h2",
                ): [(3.0 +2j, self.op1, self.op2)],
                "h2": [(4.0 -5j, self.op2)],
            }
        ).conjugate()

        targ_conjugate = MixedOp(
            {
                ("h1",): [(2.0 - 1j, self.op1.conjugate())],
                (
                    "h1",
                    "h2",
                ): [(3.0 -2j, self.op1.conjugate(), self.op2.conjugate())],
                "h2": [(4.0 +5j, self.op2.conjugate())],
            }
        )
        self.assertEqual(test_conjugate, targ_conjugate)

    def test_transpose(self):
        """Test transpose method"""
        test_transpose = MixedOp(
            {
                ("h1",): [(2.0 + 1j, self.op1)],
                (
                    "h1",
                    "h2",
                ): [(3.0 +2j, self.op1, self.op2)],
                "h2": [(4.0 -5j, self.op2)],
            }
        ).transpose()

        targ_transpose = MixedOp(
            {
                ("h1",): [(2.0 + 1j, self.op1.transpose())],
                (
                    "h1",
                    "h2",
                ): [(3.0 + 2j, self.op1.transpose(), self.op2.transpose())],
                "h2": [(4.0 - 5j, self.op2.transpose())],
            }
        )
        self.assertEqual(test_transpose, targ_transpose)

if __name__ == "__main__":
    unittest.main()
