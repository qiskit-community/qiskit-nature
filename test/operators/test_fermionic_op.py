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

"""Test for FermionicOp"""

import unittest
from itertools import product
from test import QiskitNatureTestCase

from ddt import data, ddt, unpack

from qiskit_nature import QiskitNatureError
from qiskit_nature.operators import FermionicOp


def fermion_labels(length):
    """Generate list of fermion labels with given length."""
    return [
        "".join(label) for label in product(["I", "+", "-", "N", "E"], repeat=length)
    ]


@ddt
class TestFermionicOp(QiskitNatureTestCase):
    """FermionicOp tests."""

    @data(*fermion_labels(1), *fermion_labels(2))
    def test_init(self, label):
        """Test __init__"""
        self.assertListEqual(FermionicOp(label).to_list(), [(label, 1)])

    def test_init_invalid_label(self):
        """Test __init__ with invalid label"""
        with self.assertRaises(QiskitNatureError):
            FermionicOp("INX")
        with self.assertRaises(QiskitNatureError):
            FermionicOp([("++", 1), ("EF", 1)])

    def test_init_multiterm(self):
        """Test __init__ with multi terms"""
        labels = [("N", 2), ("-", 3.14)]
        self.assertListEqual(FermionicOp(labels).to_list(), labels)

    def test_init_invalid(self):
        """Test invalid __init__"""
        with self.assertRaises(QiskitNatureError):
            FermionicOp("test")

    def test_neg(self):
        """Test __neg__"""
        fer_op = -FermionicOp("+N-EII")
        self.assertListEqual(fer_op.to_list(), [("+N-EII", -1)])

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        fer_op = FermionicOp("+-") * 2
        self.assertEqual(fer_op.to_list(), [("+-", 2)])

        fer_op = (2 + 1j) * FermionicOp([("+N", 3), ("E-", 1)])
        self.assertEqual(fer_op.to_list(), [("+N", (6 + 3j)), ("E-", (2 + 1j))])

    def test_div(self):
        """Test __truediv__"""
        fer_op = FermionicOp([("+N", 3), ("E-", 1)]) / 3
        self.assertEqual(fer_op.to_list(), [("+N", 1.0), ("E-", 0.3333333333333333)])

    def test_add(self):
        """Test __add__"""
        fer_op = 3 * FermionicOp("+N") + FermionicOp("E-")
        self.assertListEqual(fer_op.to_list(), [("+N", 3), ("E-", 1)])

        fer_op = sum(FermionicOp(label) for label in ['NIII', 'INII', 'IINI', 'IIIN'])
        self.assertListEqual(fer_op.to_list(), [('NIII', 1), ('INII', 1), ('IINI', 1), ('IIIN', 1)])

    def test_sub(self):
        """Test __sub__"""
        fer_op = 3 * FermionicOp("++") - 2 * FermionicOp("--")
        self.assertListEqual(fer_op.to_list(), [("++", 3), ("--", -2)])

    @data(*product(fermion_labels(1), fermion_labels(1)))
    @unpack
    def test_matmul(self, label1, label2):
        """Test matrix multiplication"""
        fer_op = FermionicOp(label1) @ FermionicOp(label2)
        mapping = {
            "II": "I",
            "I+": "+",
            "I-": "-",
            "IN": "N",
            "IE": "E",
            "+I": "+",
            "++": 0,
            "+-": "N",
            "+N": 0,
            "+E": "+",
            "-I": "-",
            "-+": "E",
            "--": 0,
            "-N": "-",
            "-E": 0,
            "NI": "N",
            "N+": "+",
            "N-": 0,
            "NN": "N",
            "NE": 0,
            "EI": "E",
            "E+": 0,
            "E-": "-",
            "EN": 0,
            "EE": "E",
        }
        result = mapping[label1 + label2]
        expected = [(result, 1)] if result != 0 else [("I", 0)]
        self.assertListEqual(fer_op.to_list(), expected)

    def test_matmul_multi(self):
        """Test matrix multiplication"""
        fer_op = FermionicOp("+-") @ FermionicOp("-I")
        self.assertListEqual(fer_op.to_list(), [("N-", -1)])

        fer_op = (FermionicOp("+N") + FermionicOp("E-")) @ (
            FermionicOp("II") + FermionicOp("-+")
        )
        self.assertListEqual(
            fer_op.to_list(), [("+N", 1), ("N+", 1), ("E-", 1), ("-E", -1)]
        )

    def test_pow(self):
        """Test __pow__"""
        fer_op = FermionicOp([("+N", 3), ("E-", 1)]) ** 2
        self.assertListEqual(fer_op.to_list(), [("II", 0)])

        fer_op = FermionicOp([("+N", 3), ("N-", 1)]) ** 2
        self.assertListEqual(fer_op.to_list(), [("+-", -3)])

        fer_op = (3 * FermionicOp("IIII")) ** 3
        self.assertListEqual(fer_op.to_list(), [("IIII", 27)])

        fer_op = FermionicOp([("+N", 3), ("E-", 1)]) ** 0
        self.assertListEqual(fer_op.to_list(), [("II", 1)])

    def test_adjoint(self):
        """Test adjoint method and dagger property"""
        fer_op = FermionicOp([("+N", 3), ("N-", 1), ("--", 2 + 4j)]).adjoint()
        self.assertListEqual(
            fer_op.to_list(), [("-N", 3), ("N+", 1), ("++", (-2 + 4j))]
        )

        fer_op = FermionicOp([("+-", 1), ("II", 2j)]).dagger
        self.assertListEqual(
            fer_op.to_list(), [('-+', -1), ('II', -2j)]
        )

    def test_reduce(self):
        """Test reduce"""
        fer_op = FermionicOp("N") + FermionicOp("E") + FermionicOp("N")
        reduced_op = fer_op.reduce()
        self.assertSetEqual(frozenset(reduced_op.to_list()), frozenset([("N", 2), ("E", 1)]))

        fer_op = FermionicOp(("+", 1)) + FermionicOp(("-", 1j)) + FermionicOp(("+", 1j))
        reduced_op = fer_op.reduce()
        self.assertSetEqual(frozenset(reduced_op.to_list()), frozenset([("+", 1+1j), ("-", 1j)]))

if __name__ == "__main__":
    unittest.main()
