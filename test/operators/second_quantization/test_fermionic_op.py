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
from functools import lru_cache
from itertools import product
from test import QiskitNatureTestCase

from ddt import data, ddt, unpack

from qiskit_nature.operators.second_quantization import FermionicOp

from .utils import str2list, str2str, str2tuple


@lru_cache(3)
def dense_labels(length):
    """Generate list of fermion labels with given length."""
    return ["".join(label) for label in product(["I", "+", "-", "N", "E"], repeat=length)]


@lru_cache(3)
def sparse_labels(length):
    """Generate list of fermion labels with given length."""
    return [
        " ".join(f"{char}_{i}" for i, char in enumerate(label))
        for label in product(["I", "+", "-", "N", "E"], repeat=length)
    ]


@ddt
class TestFermionicOp(QiskitNatureTestCase):
    """FermionicOp tests."""

    def assertFermionEqual(self, first: FermionicOp, second: FermionicOp):
        """Fail if two FermionicOps are different.
        Note that this equality check is approximated since the true equality check is costly.
        """
        self.assertSetEqual(frozenset(first.to_list()), frozenset(second.to_list()))

    @data(
        *product(
            (*dense_labels(1), *dense_labels(2), *dense_labels(3)),
            (str2str, str2tuple, str2list),
        )
    )
    @unpack
    def test_init(self, label, pre_processing):
        """Test __init__"""
        fer_op = FermionicOp(pre_processing(label))
        self.assertListEqual(fer_op.to_list(), [(label, 1)])
        self.assertFermionEqual(eval(repr(fer_op)), fer_op)  # pylint: disable=eval-used

    @data(
        *product(
            (
                *zip(dense_labels(1), sparse_labels(1)),
                *zip(dense_labels(2), sparse_labels(2)),
                *zip(dense_labels(3), sparse_labels(3)),
            ),
            (str2str, str2tuple, str2list),
        )
    )
    @unpack
    def test_init_sparse_label(self, labels, pre_processing):
        """Test __init__ with sparse label"""
        dense_label, sparse_label = labels
        fer_op = FermionicOp(pre_processing(sparse_label), register_length=len(dense_label))
        targ = FermionicOp(dense_label)
        self.assertFermionEqual(fer_op, targ)

    @data(
        ("INX", None),
        ([("++", 1), ("EF", 1)], None),
        ("", None),
        ("+_2 -_0", 3),
        ("+_0 -_1 +_2 -_2", 4),
    )
    @unpack
    def test_init_invalid_label(self, label, register_length):
        """Test __init__ with invalid label"""
        with self.assertRaises(ValueError):
            FermionicOp(label, register_length=register_length)

    def test_init_multiterm(self):
        """Test __init__ with multi terms"""
        with self.subTest("Test 1"):
            labels = [("N", 2), ("-", 3.14)]
            self.assertListEqual(FermionicOp(labels).to_list(), labels)

        with self.subTest("Test 2"):
            labels = [("+-", 1), ("-+", -1)]
            op = FermionicOp([("+_0 -_1", 1.0), ("-_0 +_1", -1.0)], register_length=2)
            self.assertListEqual(op.to_list(), labels)

    def test_init_multiple_digits(self):
        """Test __init__ for sparse label with multiple digits"""
        actual = FermionicOp([("-_2 +_10", 1 + 2j), ("-_12", 56)], register_length=13)
        desired = [
            ("II-IIIIIII+II", 1 + 2j),
            ("IIIIIIIIIIII-", 56),
        ]
        self.assertListEqual(actual.to_list(), desired)

    @data(str2str, str2tuple, str2list)
    def test_init_empty_str(self, pre_processing):
        """Test __init__ with empty string"""
        actual = FermionicOp(pre_processing(""), register_length=3)
        desired = FermionicOp("III")
        self.assertFermionEqual(actual, desired)

    def test_neg(self):
        """Test __neg__"""
        fer_op = -FermionicOp("+N-EII")
        targ = FermionicOp([("+N-EII", -1)])
        self.assertFermionEqual(fer_op, targ)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        with self.subTest("rightmul"):
            fer_op = FermionicOp("+-") * 2
            targ = FermionicOp([("+-", 2)])
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("left mul"):
            fer_op = (2 + 1j) * FermionicOp([("+N", 3), ("E-", 1)])
            targ = FermionicOp([("+N", (6 + 3j)), ("E-", (2 + 1j))])
            self.assertFermionEqual(fer_op, targ)

    def test_div(self):
        """Test __truediv__"""
        fer_op = FermionicOp([("+N", 3), ("E-", 1)]) / 3
        targ = FermionicOp([("+N", 1.0), ("E-", 1 / 3)])
        self.assertFermionEqual(fer_op, targ)

    def test_add(self):
        """Test __add__"""
        fer_op = 3 * FermionicOp("+N") + FermionicOp("E-")
        targ = FermionicOp([("+N", 3), ("E-", 1)])
        self.assertFermionEqual(fer_op, targ)

        fer_op = sum(FermionicOp(label) for label in ["NIII", "INII", "IINI", "IIIN"])
        targ = FermionicOp([("NIII", 1), ("INII", 1), ("IINI", 1), ("IIIN", 1)])
        self.assertFermionEqual(fer_op, targ)

    def test_sub(self):
        """Test __sub__"""
        fer_op = 3 * FermionicOp("++") - 2 * FermionicOp("--")
        targ = FermionicOp([("++", 3), ("--", -2)])
        self.assertFermionEqual(fer_op, targ)

    @data(*product(dense_labels(1), dense_labels(1)))
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
        targ = FermionicOp([(result, 1)] if result != 0 else [("I", 0)])
        self.assertFermionEqual(fer_op, targ)

    def test_matmul_multi(self):
        """Test matrix multiplication"""
        with self.subTest("single matmul"):
            fer_op = FermionicOp("+-") @ FermionicOp("-I")
            targ = FermionicOp([("N-", -1)])
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("multi matmul"):
            fer_op = FermionicOp("+-") @ FermionicOp("-I")
            fer_op = (FermionicOp("+N") + FermionicOp("E-")) @ (
                FermionicOp("II") + FermionicOp("-+")
            )
            targ = FermionicOp([("+N", 1), ("N+", 1), ("E-", 1), ("-E", -1)])
            self.assertFermionEqual(fer_op, targ)

    def test_pow(self):
        """Test __pow__"""
        with self.subTest("square trivial"):
            fer_op = FermionicOp([("+N", 3), ("E-", 1)]) ** 2
            targ = FermionicOp([("II", 0)])
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("square nontrivial"):
            fer_op = FermionicOp([("+N", 3), ("N-", 1)]) ** 2
            targ = FermionicOp([("+-", -3)])
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("3rd power"):
            fer_op = (3 * FermionicOp("IIII")) ** 3
            targ = FermionicOp([("IIII", 27)])
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("0th power"):
            fer_op = FermionicOp([("+N", 3), ("E-", 1)]) ** 0
            targ = FermionicOp([("II", 1)])
            self.assertFermionEqual(fer_op, targ)

    def test_adjoint(self):
        """Test adjoint method and dagger property"""
        with self.subTest("adjoint"):
            fer_op = ~FermionicOp([("+N", 3), ("N-", 1), ("--", 2 + 4j)])
            targ = FermionicOp([("-N", 3), ("N+", 1), ("++", (-2 + 4j))])
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("dagger"):
            fer_op = FermionicOp([("+-", 1), ("II", 2j)]).dagger
            targ = FermionicOp([("-+", -1), ("II", -2j)])
            self.assertFermionEqual(fer_op, targ)

    def test_reduce(self):
        """Test reduce"""
        with self.subTest("reduce integer"):
            fer_op = FermionicOp("N") + FermionicOp("E") + FermionicOp("N")
            reduced_op = fer_op.reduce()
            targ = FermionicOp([("N", 2), ("E", 1)])
            self.assertFermionEqual(reduced_op, targ)

        with self.subTest("reduce complex"):
            fer_op = FermionicOp("+") + 1j * FermionicOp("-") + 1j * FermionicOp("+")
            reduced_op = fer_op.reduce()
            targ = FermionicOp([("+", 1 + 1j), ("-", 1j)])
            self.assertFermionEqual(reduced_op, targ)

    def test_hermiticity(self):
        """test is_hermitian"""
        with self.subTest("operator hermitian"):
            # deliberately define test operator with duplicate terms in case .adjoint() simplifies terms
            fer_op = (
                1j * FermionicOp("+-NE")
                + 1j * FermionicOp("+-NE")
                + 1j * FermionicOp("-+NE")
                + 1j * FermionicOp("-+NE")
                + FermionicOp("+-EN")
                - FermionicOp("-+EN")
            )
            self.assertTrue(fer_op.is_hermitian())

        with self.subTest("operator not hermitian"):
            fer_op = (
                1j * FermionicOp("+-NE")
                + 1j * FermionicOp("+-NE")
                - 1j * FermionicOp("-+NE")
                - 1j * FermionicOp("-+NE")
            )
            self.assertFalse(fer_op.is_hermitian())


if __name__ == "__main__":
    unittest.main()
