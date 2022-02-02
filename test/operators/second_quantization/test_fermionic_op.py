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

"""Test for FermionicOp"""

import unittest
import warnings
from functools import lru_cache
from itertools import product
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt, unpack
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs

from qiskit_nature.operators.second_quantization import FermionicOp

from .utils import str2list, str2str, str2tuple


@lru_cache(3)
def dense_labels(length):
    """Generate list of fermion labels with given length."""
    return ["".join(label) for label in product(["I", "+", "-", "N", "E"], repeat=length)]


@lru_cache(3)
def sparse_labels(length, only_plus_minus=False):
    """Generate list of fermion labels with given length."""
    generator = ["+", "-"] if only_plus_minus else ["I", "+", "-", "N", "E"]
    return [
        " ".join(f"{char}_{i}" for i, char in enumerate(label))
        for label in product(generator, repeat=length)
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
        fer_op = FermionicOp(pre_processing(label), display_format="dense")
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
        fer_op = FermionicOp(
            pre_processing(sparse_label), register_length=len(dense_label), display_format="sparse"
        )
        targ = FermionicOp(dense_label, display_format="sparse")
        self.assertFermionEqual(fer_op, targ)

    @data(
        ("INX", None),
        ([("++", 1), ("EF", 1)], None),
    )
    @unpack
    def test_init_invalid_label(self, label, register_length):
        """Test __init__ with invalid label"""
        with self.assertRaises(ValueError):
            FermionicOp(label, register_length=register_length, display_format="dense")

    def test_init_multiterm(self):
        """Test __init__ with multi terms"""
        with self.subTest("Test 1"):
            labels = [("N", 2), ("-", 3.14)]
            self.assertListEqual(FermionicOp(labels, display_format="dense").to_list(), labels)

        with self.subTest("Test 2"):
            labels = [("+-", 1), ("-+", -1)]
            op = FermionicOp(
                [("+_0 -_1", 1.0), ("-_0 +_1", -1.0)], register_length=2, display_format="dense"
            )
            self.assertListEqual(op.to_list(), labels)

    def test_init_multiple_digits(self):
        """Test __init__ for sparse label with multiple digits"""
        actual = FermionicOp(
            [("-_2 +_10", 1 + 2j), ("-_12", 56)], register_length=13, display_format="dense"
        )
        desired = [
            ("II-IIIIIII+II", 1 + 2j),
            ("IIIIIIIIIIII-", 56),
        ]
        self.assertListEqual(actual.to_list(), desired)

    @data(str2str, str2tuple, str2list)
    def test_init_empty_str(self, pre_processing):
        """Test __init__ with empty string"""
        actual = FermionicOp(pre_processing(""), register_length=3, display_format="dense")
        desired = FermionicOp("III", display_format="dense")
        self.assertFermionEqual(actual, desired)

    def test_init_from_tuple_label(self):
        """Test __init__ for tuple"""
        actual = FermionicOp(
            [([(0, 2), (1, 10)], 1 + 2j), ([(0, 12)], 56)],
            register_length=13,
            display_format="dense",
        )
        desired = [
            ("II-IIIIIII+II", 1 + 2j),
            ("IIIIIIIIIIII-", 56),
        ]
        self.assertListEqual(actual.to_list(), desired)

    def test_register_length(self):
        """Test inference of register_length"""
        op = FermionicOp([("+_1", 1.0), ("", 1.0)], display_format="dense")
        self.assertEqual(op.register_length, 2)

    def test_neg(self):
        """Test __neg__"""
        fer_op = -FermionicOp("+N-EII", display_format="dense")
        targ = FermionicOp([("+N-EII", -1)], display_format="dense")
        self.assertFermionEqual(fer_op, targ)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        with self.subTest("rightmul"):
            fer_op = FermionicOp("+-", display_format="dense") * 2
            targ = FermionicOp([("+-", 2)], display_format="dense")
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("left mul"):
            fer_op = (2 + 1j) * FermionicOp([("+N", 3), ("E-", 1)], display_format="dense")
            targ = FermionicOp([("+N", (6 + 3j)), ("E-", (2 + 1j))], display_format="dense")
            self.assertFermionEqual(fer_op, targ)

    def test_div(self):
        """Test __truediv__"""
        fer_op = FermionicOp([("+N", 3), ("E-", 1)], display_format="dense") / 3
        targ = FermionicOp([("+N", 1.0), ("E-", 1 / 3)], display_format="dense")
        self.assertFermionEqual(fer_op, targ)

    def test_add(self):
        """Test __add__"""
        fer_op = 3 * FermionicOp("+N", display_format="dense") + FermionicOp(
            "E-", display_format="dense"
        )
        targ = FermionicOp([("+N", 3), ("E-", 1)], display_format="dense")
        self.assertFermionEqual(fer_op, targ)

        fer_op = sum(
            FermionicOp(label, display_format="dense") for label in ["NIII", "INII", "IINI", "IIIN"]
        )
        targ = FermionicOp(
            [("NIII", 1), ("INII", 1), ("IINI", 1), ("IIIN", 1)], display_format="dense"
        )
        self.assertFermionEqual(fer_op, targ)

    def test_sub(self):
        """Test __sub__"""
        fer_op = 3 * FermionicOp("++", display_format="dense") - 2 * FermionicOp(
            "--", display_format="dense"
        )
        targ = FermionicOp([("++", 3), ("--", -2)], display_format="dense")
        self.assertFermionEqual(fer_op, targ)

    @data(*product(dense_labels(1), dense_labels(1)))
    @unpack
    def test_matmul(self, label1, label2):
        """Test matrix multiplication"""
        fer_op = FermionicOp(label1, display_format="dense") @ FermionicOp(
            label2, display_format="dense"
        )
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
        targ = FermionicOp([(result, 1)] if result != 0 else [("I", 0)], display_format="dense")
        self.assertFermionEqual(fer_op, targ)

    def test_matmul_multi(self):
        """Test matrix multiplication"""
        with self.subTest("single matmul"):
            fer_op = FermionicOp("+-", display_format="dense") @ FermionicOp(
                "-I", display_format="dense"
            )
            targ = FermionicOp([("N-", -1)], display_format="dense")
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("multi matmul"):
            fer_op = FermionicOp("+-", display_format="dense") @ FermionicOp(
                "-I", display_format="dense"
            )
            fer_op = (
                FermionicOp("+N", display_format="dense")
                + FermionicOp("E-", display_format="dense")
            ) @ (
                FermionicOp("II", display_format="dense")
                + FermionicOp("-+", display_format="dense")
            )
            targ = FermionicOp(
                [("+N", 1), ("N+", 1), ("E-", 1), ("-E", -1)], display_format="dense"
            )
            self.assertFermionEqual(fer_op, targ)

    def test_pow(self):
        """Test __pow__"""
        with self.subTest("square trivial"):
            fer_op = FermionicOp([("+N", 3), ("E-", 1)], display_format="dense") ** 2
            targ = FermionicOp([("II", 0)], display_format="dense")
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("square nontrivial"):
            fer_op = FermionicOp([("+N", 3), ("N-", 1)], display_format="dense") ** 2
            targ = FermionicOp([("+-", -3)], display_format="dense")
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("3rd power"):
            fer_op = (3 * FermionicOp("IIII", display_format="dense")) ** 3
            targ = FermionicOp([("IIII", 27)], display_format="dense")
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("0th power"):
            fer_op = FermionicOp([("+N", 3), ("E-", 1)], display_format="dense") ** 0
            targ = FermionicOp([("II", 1)], display_format="dense")
            self.assertFermionEqual(fer_op, targ)

    def test_adjoint(self):
        """Test adjoint method"""
        with self.subTest("adjoint"):
            fer_op = ~FermionicOp([("+N", 3), ("N-", 1), ("--", 2 + 4j)], display_format="dense")
            targ = FermionicOp([("-N", 3), ("N+", 1), ("++", (-2 + 4j))], display_format="dense")
            self.assertFermionEqual(fer_op, targ)

        with self.subTest("adjoint 2"):
            fer_op = FermionicOp([("+-", 1), ("II", 2j)], display_format="dense").adjoint()
            targ = FermionicOp([("-+", -1), ("II", -2j)], display_format="dense")
            self.assertFermionEqual(fer_op, targ)

    def test_reduce(self):
        """Test reduce"""
        with self.subTest("reduce integer"):
            fer_op = (
                FermionicOp("N", display_format="dense")
                + FermionicOp("E", display_format="dense")
                + FermionicOp("N", display_format="dense")
            )
            reduced_op = fer_op.reduce()
            targ = FermionicOp([("N", 1), ("I", 1)], display_format="dense")
            self.assertFermionEqual(reduced_op, targ)

        with self.subTest("reduce complex"):
            fer_op = (
                FermionicOp("+", display_format="dense")
                + 1j * FermionicOp("-", display_format="dense")
                + 1j * FermionicOp("+", display_format="dense")
            )
            reduced_op = fer_op.reduce()
            targ = FermionicOp([("+", 1 + 1j), ("-", 1j)], display_format="dense")
            self.assertFermionEqual(reduced_op, targ)

    def test_hermiticity(self):
        """test is_hermitian"""
        with self.subTest("operator hermitian"):
            # deliberately define test operator with duplicate terms in case .adjoint() simplifies terms
            fer_op = (
                1j * FermionicOp("+-NE", display_format="dense")
                + 1j * FermionicOp("+-NE", display_format="dense")
                + 1j * FermionicOp("-+NE", display_format="dense")
                + 1j * FermionicOp("-+NE", display_format="dense")
                + FermionicOp("+-EN", display_format="dense")
                - FermionicOp("-+EN", display_format="dense")
            )
            self.assertTrue(fer_op.is_hermitian())

        with self.subTest("operator not hermitian"):
            fer_op = (
                1j * FermionicOp("+-NE", display_format="dense")
                + 1j * FermionicOp("+-NE", display_format="dense")
                - 1j * FermionicOp("-+NE", display_format="dense")
                - 1j * FermionicOp("-+NE", display_format="dense")
            )
            self.assertFalse(fer_op.is_hermitian())

    @data(
        *product(
            (
                *sparse_labels(1, True),
                *sparse_labels(2, True),
                *sparse_labels(3, True),
            ),
            (str2str, str2tuple, str2list),
        )
    )
    @unpack
    def test_label_display_mode(self, label, pre_processing):
        """test label_display_mode"""
        fer_op = FermionicOp(pre_processing(label), display_format="dense")

        fer_op.display_format = "sparse"
        self.assertListEqual(fer_op.to_list(), str2list(label))
        fer_op.display_format = "dense"
        self.assertNotEqual(fer_op.to_list(), str2list(label))

    def test_to_matrix(self):
        """Test to_matrix"""
        with self.subTest("identity operator matrix"):
            mat = FermionicOp.one(2).to_matrix(sparse=False)
            targ = np.eye(4)
            self.assertTrue(np.allclose(mat, targ))

        with self.subTest("number operator matrix"):
            mat = FermionicOp("IN").to_matrix(sparse=False)
            targ = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
            self.assertTrue(np.allclose(mat, targ))

        with self.subTest("emptiness operator matrix"):
            mat = FermionicOp("IE").to_matrix(sparse=False)
            targ = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
            self.assertTrue(np.allclose(mat, targ))

        with self.subTest("raising operator matrix"):
            mat = FermionicOp("I+").to_matrix(sparse=False)
            targ = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0]])
            self.assertTrue(np.allclose(mat, targ))

        with self.subTest("lowering operator matrix"):
            mat = FermionicOp("I-").to_matrix(sparse=False)
            targ = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0]])
            self.assertTrue(np.allclose(mat, targ))

        with self.subTest("nontrivial sparse matrix"):
            mat = FermionicOp([("ENI+", 3j), ("-N+-", -2)]).to_matrix()
            targ = csc_matrix(([-3j, 3j, -2], ([5, 7, 6], [4, 6, 13])), shape=(16, 16))
            self.assertTrue((mat != targ).nnz == 0)

        with self.subTest("Test Hydrogen spectrum"):
            h2_labels = [
                ("+_0 -_1 +_2 -_3", (0.18093120148374142)),
                ("+_0 -_1 -_2 +_3", (-0.18093120148374134)),
                ("-_0 +_1 +_2 -_3", (-0.18093120148374134)),
                ("-_0 +_1 -_2 +_3", (0.18093120148374128)),
                ("+_3 -_3", (-0.4718960038869427)),
                ("+_2 -_2", (-1.2563391028292563)),
                ("+_2 -_2 +_3 -_3", (0.48365053378098793)),
                ("+_1 -_1", (-0.4718960038869427)),
                ("+_1 -_1 +_3 -_3", (0.6985737398458793)),
                ("+_1 -_1 +_2 -_2", (0.6645817352647293)),
                ("+_0 -_0", (-1.2563391028292563)),
                ("+_0 -_0 +_3 -_3", (0.6645817352647293)),
                ("+_0 -_0 +_2 -_2", (0.6757101625347564)),
                ("+_0 -_0 +_1 -_1", (0.48365053378098793)),
            ]
            h2_matrix = FermionicOp(h2_labels, register_length=4).to_matrix()
            evals, evecs = eigs(h2_matrix)
            self.assertTrue(np.isclose(np.min(evals), -1.8572750))
            # make sure the ground state has support only in the 2-particle subspace
            groundstate = evecs[:, np.argmin(evals)]
            for idx in np.where(~np.isclose(groundstate, 0))[0]:
                binary = f"{idx:0{4}b}"
                self.assertEqual(binary.count("1"), 2)

    def test_normal_order(self):
        """test normal_order method"""
        with self.subTest("Test for creation operator"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                orig = FermionicOp("+")
            fer_op = orig.to_normal_order()
            targ = FermionicOp("+_0", display_format="sparse")
            self.assertFermionEqual(fer_op, targ)
            self.assertEqual(orig.display_format, "dense")

        with self.subTest("Test for annihilation operator"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                orig = FermionicOp("-")
            fer_op = orig.to_normal_order()
            targ = FermionicOp("-_0", display_format="sparse")
            self.assertFermionEqual(fer_op, targ)
            self.assertEqual(orig.display_format, "dense")

        with self.subTest("Test for number operator"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                orig = FermionicOp("N")
            fer_op = orig.to_normal_order()
            targ = FermionicOp("+_0 -_0", display_format="sparse")
            self.assertFermionEqual(fer_op, targ)
            self.assertEqual(orig.display_format, "dense")

        with self.subTest("Test for empty operator"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                orig = FermionicOp("E")
            fer_op = orig.to_normal_order()
            targ = FermionicOp([("", 1), ("+_0 -_0", -1)], display_format="sparse")
            self.assertFermionEqual(fer_op, targ)
            self.assertEqual(orig.display_format, "dense")

        with self.subTest("Test for multiple operators 1"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                orig = FermionicOp("-+")
            fer_op = orig.to_normal_order()
            targ = FermionicOp([("+_1 -_0", -1)], display_format="sparse")
            self.assertFermionEqual(fer_op, targ)
            self.assertEqual(orig.display_format, "dense")

        with self.subTest("Test for multiple operators 2"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                orig = 3 * FermionicOp("E+-")
            fer_op = orig.to_normal_order()
            targ = FermionicOp([("+_1 -_2", 3), ("+_0 +_1 -_0 -_2", 3)], display_format="sparse")
            self.assertFermionEqual(fer_op, targ)
            self.assertEqual(orig.display_format, "dense")


if __name__ == "__main__":
    unittest.main()
