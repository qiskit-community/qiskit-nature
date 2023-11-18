# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for MajoranaOp"""

import unittest

from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt, unpack
from qiskit.circuit import Parameter

from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.operators import MajoranaOp, FermionicOp, PolynomialTensor
import qiskit_nature.optionals as _optionals


@ddt
class TestMajoranaOp(QiskitNatureTestCase):
    """MajoranaOp tests."""

    a = Parameter("a")
    b = Parameter("b")

    op1 = MajoranaOp({"_0 _1": 1})
    op2 = MajoranaOp({"_1 _0": 2})
    op3 = MajoranaOp({"_0 _1": 1, "_1 _0": 2})
    op4 = MajoranaOp({"_0 _1": a})

    def test_neg(self):
        """Test __neg__"""
        maj_op = -self.op1
        targ = MajoranaOp({"_0 _1": -1}, num_spin_orbitals=1)
        self.assertEqual(maj_op, targ)

        maj_op = -self.op4
        targ = MajoranaOp({"_0 _1": -self.a})
        self.assertEqual(maj_op, targ)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        with self.subTest("rightmul"):
            maj_op = self.op1 * 2
            targ = MajoranaOp({"_0 _1": 2}, num_spin_orbitals=1)
            self.assertEqual(maj_op, targ)

            maj_op = self.op1 * self.a
            targ = MajoranaOp({"_0 _1": self.a})
            self.assertEqual(maj_op, targ)

        with self.subTest("left mul"):
            maj_op = (2 + 1j) * self.op3
            targ = MajoranaOp({"_0 _1": (2 + 1j), "_1 _0": (4 + 2j)}, num_spin_orbitals=1)
            self.assertEqual(maj_op, targ)

    def test_div(self):
        """Test __truediv__"""
        maj_op = self.op1 / 2
        targ = MajoranaOp({"_0 _1": 0.5}, num_spin_orbitals=1)
        self.assertEqual(maj_op, targ)

        maj_op = self.op1 / self.a
        targ = MajoranaOp({"_0 _1": 1 / self.a})
        self.assertEqual(maj_op, targ)

    def test_add(self):
        """Test __add__"""
        maj_op = self.op1 + self.op2
        targ = self.op3
        self.assertEqual(maj_op, targ)

        maj_op = self.op1 + self.op4
        targ = MajoranaOp({"_0 _1": 1 + self.a})
        self.assertEqual(maj_op, targ)

        with self.subTest("sum"):
            maj_op = sum(MajoranaOp({label: 1}) for label in ["_0", "_1", "_2 _3"])
            targ = MajoranaOp({"_0": 1, "_1": 1, "_2 _3": 1})
            self.assertEqual(maj_op, targ)

    def test_sub(self):
        """Test __sub__"""
        maj_op = self.op3 - self.op2
        targ = MajoranaOp({"_0 _1": 1, "_1 _0": 0}, num_spin_orbitals=1)
        self.assertEqual(maj_op, targ)

        maj_op = self.op4 - self.op1
        targ = MajoranaOp({"_0 _1": self.a - 1})
        self.assertEqual(maj_op, targ)

    def test_compose(self):
        """Test operator composition"""
        with self.subTest("single compose"):
            maj_op = MajoranaOp({"_0 _2": 1}, num_spin_orbitals=2) @ MajoranaOp(
                {"_1": 1}, num_spin_orbitals=2
            )
            targ = MajoranaOp({"_0 _2 _1": 1}, num_spin_orbitals=2)
            self.assertEqual(maj_op, targ)

        with self.subTest("single compose with parameters"):
            maj_op = MajoranaOp({"_0 _2": self.a}) @ MajoranaOp({"_1": 1})
            targ = MajoranaOp({"_0 _2 _1": self.a})
            self.assertEqual(maj_op, targ)

        with self.subTest("multi compose"):
            maj_op = MajoranaOp({"_0 _2 _3": 1, "_1 _2 _3": 1}, num_spin_orbitals=2) @ MajoranaOp(
                {"": 1, "_1 _3": 1}, num_spin_orbitals=2
            )
            maj_op = maj_op.simplify()
            targ = MajoranaOp(
                {"_0 _2 _3": 1, "_1 _2 _3": 1, "_0 _2 _1": -1, "_2": 1},
                num_spin_orbitals=2,
            )
            self.assertEqual(maj_op, targ)

        with self.subTest("multi compose with parameters"):
            maj_op = MajoranaOp({"_0 _2 _3": self.a, "_1 _0 _3": 1}) @ MajoranaOp(
                {"": 1, "_0 _3": self.b}
            )
            maj_op = maj_op.simplify()
            targ = MajoranaOp(
                {
                    "_0 _2 _3": self.a,
                    "_1 _0 _3": 1,
                    "_2": self.a * self.b,
                    "_1": -self.b,
                }
            )
            self.assertEqual(maj_op, targ)

    def test_tensor(self):
        """Test tensor multiplication"""
        maj_op = self.op1.tensor(self.op2)
        targ = MajoranaOp({"_0 _1 _3 _2": 2}, num_spin_orbitals=2)
        self.assertEqual(maj_op, targ)

        maj_op = self.op4.tensor(self.op2)
        targ = MajoranaOp({"_0 _1 _3 _2": 2 * self.a})
        self.assertEqual(maj_op, targ)

    def test_expand(self):
        """Test reversed tensor multiplication"""
        maj_op = self.op1.expand(self.op2)
        targ = MajoranaOp({"_1 _0 _2 _3": 2}, num_spin_orbitals=2)
        self.assertEqual(maj_op, targ)

        maj_op = self.op4.expand(self.op2)
        targ = MajoranaOp({"_1 _0 _2 _3": 2 * self.a})
        self.assertEqual(maj_op, targ)

    def test_pow(self):
        """Test __pow__"""
        with self.subTest("square"):
            maj_op = MajoranaOp({"_0 _1 _2": 3, "_1 _0 _3": 1}, num_spin_orbitals=2) ** 2
            maj_op = maj_op.simplify()
            targ = MajoranaOp({"": -10, "_2 _3": 3, "_3 _2": 3}, num_spin_orbitals=2)
            self.assertEqual(maj_op, targ)

        with self.subTest("3rd power"):
            maj_op = (3 * MajoranaOp.one()) ** 3
            targ = 27 * MajoranaOp.one()
            self.assertEqual(maj_op, targ)

        with self.subTest("0th power"):
            maj_op = MajoranaOp({"_0 _1 _2": 3, "_1 _0 _3": 1}, num_spin_orbitals=2) ** 0
            maj_op = maj_op.simplify()
            targ = MajoranaOp.one()
            self.assertEqual(maj_op, targ)

        with self.subTest("square with parameters"):
            maj_op = MajoranaOp({"_0 _1 _2": self.a, "_1 _0 _3": 1}, num_spin_orbitals=2) ** 2
            maj_op = maj_op.simplify()
            square = (2 * self.a.log()).exp()  # qiskit.circuit.Parameter has no pow method
            targ = MajoranaOp(
                {"": -1 - square, "_2 _3": self.a, "_3 _2": self.a}, num_spin_orbitals=2
            )
            self.assertEqual(maj_op, targ)

    def test_adjoint(self):
        """Test adjoint method"""
        maj_op = MajoranaOp(
            {"": 1j, "_0 _1 _2": 3, "_0 _1 _3": 1, "_1 _3": 2 + 4j}, num_spin_orbitals=3
        ).adjoint()
        targ = MajoranaOp(
            {"": -1j, "_2 _1 _0": 3, "_3 _1 _0": 1, "_3 _1": 2 - 4j}, num_spin_orbitals=3
        )
        self.assertEqual(maj_op, targ)

        maj_op = MajoranaOp(
            {"": 1j, "_0 _1 _2": 3, "_0 _1 _3": self.a, "_1 _3": 2 + 4j}, num_spin_orbitals=3
        ).adjoint()
        targ = MajoranaOp(
            {"": -1j, "_2 _1 _0": 3, "_3 _1 _0": self.a.conjugate(), "_3 _1": 2 - 4j},
            num_spin_orbitals=3,
        )
        self.assertEqual(maj_op, targ)

    def test_simplify(self):
        """Test simplify"""
        with self.subTest("simplify integer"):
            maj_op = MajoranaOp({"_0 _1": 1, "_0 _1 _1 _1": 1}, num_spin_orbitals=1)
            simplified_op = maj_op.simplify()
            targ = MajoranaOp({"_0 _1": 2}, num_spin_orbitals=1)
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify complex"):
            maj_op = MajoranaOp({"_0 _1": 1, "_0 _1 _0 _0": 1j}, num_spin_orbitals=1)
            simplified_op = maj_op.simplify()
            targ = MajoranaOp({"_0 _1": 1 + 1j}, num_spin_orbitals=1)
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify doesn't reorder"):
            maj_op = MajoranaOp({"_1 _2": 1 + 0j}, num_spin_orbitals=2)
            simplified_op = maj_op.simplify()
            self.assertEqual(simplified_op, maj_op)

            maj_op = MajoranaOp({"_3 _0": 1 + 0j}, num_spin_orbitals=2)
            simplified_op = maj_op.simplify()
            self.assertEqual(simplified_op, maj_op)

        with self.subTest("simplify zero"):
            maj_op = self.op1 - self.op1
            simplified_op = maj_op.simplify()
            targ = MajoranaOp.zero()
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify parameters"):
            maj_op = MajoranaOp({"_0 _1": self.a, "_0 _1 _0 _0": 1j})
            simplified_op = maj_op.simplify()
            targ = MajoranaOp({"_0 _1": self.a + 1j})
            self.assertEqual(simplified_op, targ)

        with self.subTest("simplify + index order"):
            orig = MajoranaOp({"_3 _1 _0 _1": 1, "_0 _3": 2})
            maj_op = orig.simplify().index_order()
            targ = MajoranaOp({"_0 _3": 3})
            self.assertEqual(maj_op, targ)

    def test_hermiticity(self):
        """test is_hermitian"""
        with self.subTest("operator hermitian"):
            maj_op = (
                1j * MajoranaOp({"_0 _1 _2 _3": 1}, num_spin_orbitals=2)
                - 1j * MajoranaOp({"_3 _2 _1 _0": 1}, num_spin_orbitals=2)
                + MajoranaOp({"_0 _1": 1}, num_spin_orbitals=2)
                + MajoranaOp({"_1 _0": 1}, num_spin_orbitals=2)
            )
            self.assertTrue(maj_op.is_hermitian())

        with self.subTest("operator not hermitian"):
            maj_op = (
                1j * MajoranaOp({"_0 _1 _2 _3": 1}, num_spin_orbitals=2)
                + 1j * MajoranaOp({"_3 _2 _1 _0": 1}, num_spin_orbitals=2)
                + MajoranaOp({"_0 _1": 1}, num_spin_orbitals=2)
                - MajoranaOp({"_1 _0": 1}, num_spin_orbitals=2)
            )
            self.assertFalse(maj_op.is_hermitian())

        with self.subTest("test passing atol"):
            maj_op = MajoranaOp({"_0 _1": 1}, num_spin_orbitals=2) + (1 + 1e-7) * MajoranaOp(
                {"_1 _0": 1}, num_spin_orbitals=2
            )
            self.assertFalse(maj_op.is_hermitian())
            self.assertFalse(maj_op.is_hermitian(atol=1e-8))
            self.assertTrue(maj_op.is_hermitian(atol=1e-6))

        with self.subTest("parameters"):
            maj_op = MajoranaOp({"_0": self.a})
            with self.assertRaisesRegex(ValueError, "parameter"):
                _ = maj_op.is_hermitian()

    def test_equiv(self):
        """test equiv"""
        prev_atol = MajoranaOp.atol
        prev_rtol = MajoranaOp.rtol
        op3 = self.op1 + (1 + 0.00005) * self.op2
        self.assertFalse(op3.equiv(self.op3))
        MajoranaOp.atol = 1e-4
        MajoranaOp.rtol = 1e-4
        self.assertTrue(op3.equiv(self.op3))
        MajoranaOp.atol = prev_atol
        MajoranaOp.rtol = prev_rtol

    def test_index_order(self):
        """test index_order method"""
        ordered_op = MajoranaOp({"_0 _1": 1})
        reverse_op = MajoranaOp({"_1 _0": -1})
        maj_op = ordered_op.index_order()
        self.assertEqual(maj_op, ordered_op)
        maj_op = reverse_op.index_order()
        self.assertEqual(maj_op, ordered_op)

    def test_induced_norm(self):
        """Test induced norm."""
        op1 = 3 * MajoranaOp({"_0": 1}, num_spin_orbitals=1) + 4j * MajoranaOp(
            {"_1": 1}, num_spin_orbitals=1
        )
        op2 = 3 * MajoranaOp({"_0": 1}, num_spin_orbitals=1) + 4j * MajoranaOp(
            {"_0": 1}, num_spin_orbitals=1
        )
        self.assertAlmostEqual(op1.induced_norm(), 7.0)
        self.assertAlmostEqual(op1.induced_norm(2), 5.0)
        self.assertAlmostEqual(op2.induced_norm(), 5.0)
        self.assertAlmostEqual(op2.induced_norm(2), 5.0)

    @unpack
    @data(
        ("", 1, True),  # empty string
        ("_0", 1, True),  # single term
        ("_0 _1", 2, True),  # multiple terms
        ("_0 _3", 4, True),  # multiple orbitals
        ("_1 _1", 2, True),  # identical terms
        ("_10", 11, True),  # multiple digits
        (" _0", 1, False),  # leading whitespace
        ("_0 ", 1, False),  # trailing whitespace
        ("_0  _0", 1, False),  # multiple separating spaces
        ("_0a", 1, False),  # incorrect term pattern
        ("_a0", 1, False),  # incorrect term pattern
        ("0_", 1, False),  # incorrect term pattern
        ("+_0", 1, False),  # incorrect fermionic pattern
        ("something", 1, False),  # incorrect term pattern
        ("_1", 1, True),  # 1 spin orbital takes two registers
        ("_2", 1, False),  # register length is too short
    )
    def test_validate(self, key: str, length: int, valid: bool):
        """Test key validation."""
        num_so = (length + 1) // 2
        if valid:
            _ = MajoranaOp({key: 1.0}, num_spin_orbitals=num_so)
        else:
            with self.assertRaises(QiskitNatureError):
                _ = MajoranaOp({key: 1.0}, num_spin_orbitals=num_so)

    def test_no_copy(self):
        """Test constructor with copy=False"""
        test_dict = {"_0 _1": 1}
        op = MajoranaOp(test_dict, copy=False)
        test_dict["_0 _1"] = 2
        self.assertEqual(op, MajoranaOp({"_0 _1": 2}))

    def test_no_validate(self):
        """Test skipping validation"""
        with self.subTest("no validation"):
            op = MajoranaOp({"_0 _1": 1}, num_spin_orbitals=1, validate=False)
            self.assertEqual(op, MajoranaOp({"_0 _1": 1}))

        with self.subTest("no validation no num_spin_orbitals"):
            op = MajoranaOp({"_0 _1": 1}, validate=False)
            self.assertEqual(op.num_spin_orbitals, None)

        with self.subTest("no validation with wrong label"):
            op = MajoranaOp({"test": 1}, validate=False)
            with self.assertRaises(ValueError):
                list(op.terms())

        with self.subTest("no validation with wrong num_spin_orbitals"):
            op = MajoranaOp({"_1 _2": 1}, num_spin_orbitals=1, validate=False)
            self.assertEqual(MajoranaOp.from_terms(op.terms()).num_spin_orbitals, 2)

    def test_from_polynomial_tensor(self):
        """Test from PolynomialTensor construction"""

        with self.subTest("dense tensor"):
            p_t = PolynomialTensor(
                {
                    "_": np.arange(1, 3),
                    "__": np.arange(1, 5).reshape((2, 2)),
                }
            )
            op = MajoranaOp.from_polynomial_tensor(p_t)

            expected = MajoranaOp(
                {
                    "_0": 1,
                    "_1": 2,
                    "_0 _0": 1,
                    "_0 _1": 2,
                    "_1 _0": 3,
                    "_1 _1": 4,
                },
                num_spin_orbitals=1,
            )

            self.assertEqual(op, expected)

        if _optionals.HAS_SPARSE:
            import sparse as sp  # pyright: ignore # pylint: disable=import-error

            with self.subTest("sparse tensor"):
                r_l = 2
                p_t = PolynomialTensor(
                    {
                        "__": sp.as_coo({(0, 0): 1, (1, 0): 2}, shape=(r_l, r_l)),
                        "____": sp.as_coo(
                            {(0, 0, 0, 1): 1, (1, 0, 1, 1): 2}, shape=(r_l, r_l, r_l, r_l)
                        ),
                    }
                )
                op = MajoranaOp.from_polynomial_tensor(p_t)

                expected = MajoranaOp(
                    {
                        "_0 _0": 1,
                        "_1 _0": 2,
                        "_0 _0 _0 _1": 1,
                        "_1 _0 _1 _1": 2,
                    },
                    num_spin_orbitals=r_l,
                )

                self.assertEqual(op, expected)

        with self.subTest("compose operation order"):
            r_l = 2
            p_t = PolynomialTensor(
                {
                    "__": np.arange(1, 5).reshape((r_l, r_l)),
                    "____": np.arange(1, 17).reshape((r_l, r_l, r_l, r_l)),
                }
            )
            op = MajoranaOp.from_polynomial_tensor(p_t)

            a = op @ op
            b = MajoranaOp.from_polynomial_tensor(p_t @ p_t)
            self.assertEqual(a, b)

        with self.subTest("tensor operation order"):
            r_l = 2
            p_t = PolynomialTensor(
                {
                    "__": np.arange(1, 5).reshape((r_l, r_l)),
                    "____": np.arange(1, 17).reshape((r_l, r_l, r_l, r_l)),
                }
            )
            op = MajoranaOp.from_polynomial_tensor(p_t)

            self.assertEqual(op ^ op, MajoranaOp.from_polynomial_tensor(p_t ^ p_t))

    def test_no_num_spin_orbitals(self):
        """Test operators with automatic register length"""
        op0 = MajoranaOp({"": 1})
        op1 = MajoranaOp({"_0 _1": 1})
        op2 = MajoranaOp({"_0 _1 _2": 2})

        with self.subTest("Inferred register length"):
            self.assertEqual(op0.num_spin_orbitals, 0)
            self.assertEqual(op1.num_spin_orbitals, 1)
            self.assertEqual(op2.num_spin_orbitals, 2)

        with self.subTest("Mathematical operations"):
            self.assertEqual((op0 + op2).num_spin_orbitals, 2)
            self.assertEqual((op1 + op2).num_spin_orbitals, 2)
            self.assertEqual((op0 @ op2).num_spin_orbitals, 2)
            self.assertEqual((op1 @ op2).num_spin_orbitals, 2)
            self.assertEqual((op1 ^ op2).num_spin_orbitals, 3)

        with self.subTest("Equality"):
            op3 = MajoranaOp({"_0 _1": 1}, num_spin_orbitals=3)
            self.assertEqual(op1, op3)
            self.assertTrue(op1.equiv(1.000001 * op3))

    def test_terms(self):
        """Test terms generator."""
        op = MajoranaOp(
            {
                "_0": 1,
                "_0 _1": 2,
                "_1 _2 _3": 2,
            }
        )

        terms = [([("+", 0)], 1), ([("+", 0), ("+", 1)], 2), ([("+", 1), ("+", 2), ("+", 3)], 2)]

        with self.subTest("terms"):
            self.assertEqual(list(op.terms()), terms)

        with self.subTest("from_terms"):
            self.assertEqual(MajoranaOp.from_terms(terms), op)

    def test_permute_indices(self):
        """Test index permutation method."""
        op = MajoranaOp(
            {
                "_0 _1": 1,
                "_1 _2": 2,
            },
            num_spin_orbitals=2,
        )

        with self.subTest("wrong permutation length"):
            with self.assertRaises(ValueError):
                _ = op.permute_indices([1, 0])

        with self.subTest("actual permutation"):
            permuted_op = op.permute_indices([2, 1, 3, 0])

            self.assertEqual(permuted_op, MajoranaOp({"_2 _1": 1, "_1 _3": 2}, num_spin_orbitals=2))

    def test_reg_len_with_skipped_key_validation(self):
        """Test the behavior of `register_length` after key validation was skipped."""
        new_op = MajoranaOp({"_0 _1": 1}, validate=False)
        self.assertIsNone(new_op.num_spin_orbitals)
        self.assertEqual(new_op.register_length, 2)

    def test_from_fermionic_op(self):
        """Test conversion from FermionicOp."""
        original_ops = [
            FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2),
            FermionicOp({"+_0 -_0 +_1 -_1": 2}, num_spin_orbitals=2),
            FermionicOp({"+_0 +_1 -_2 -_1": 3}, num_spin_orbitals=3),
        ]
        expected_ops_no_simp_no_order = [
            MajoranaOp(
                {"_0 _2": 0.25, "_0 _3": 0.25j, "_1 _2": -0.25j, "_1 _3": 0.25}, num_spin_orbitals=2
            ),
            2
            * MajoranaOp(
                {
                    "_0 _0 _2 _2": 1 / 16,
                    "_0 _1 _2 _2": 1j / 16,
                    "_1 _0 _2 _2": -1j / 16,
                    "_1 _1 _2 _2": 1 / 16,
                    #
                    "_0 _0 _2 _3": 1j / 16,
                    "_0 _1 _2 _3": -1 / 16,
                    "_1 _0 _2 _3": 1 / 16,
                    "_1 _1 _2 _3": 1j / 16,
                    #
                    "_0 _0 _3 _2": -1j / 16,
                    "_0 _1 _3 _2": 1 / 16,
                    "_1 _0 _3 _2": -1 / 16,
                    "_1 _1 _3 _2": -1j / 16,
                    #
                    "_0 _0 _3 _3": 1 / 16,
                    "_0 _1 _3 _3": 1j / 16,
                    "_1 _0 _3 _3": -1j / 16,
                    "_1 _1 _3 _3": 1 / 16,
                },
                num_spin_orbitals=2,
            ),
            3
            * MajoranaOp(
                {
                    "_0 _2 _4 _2": 1 / 16,
                    "_0 _3 _4 _2": -1j / 16,
                    "_1 _2 _4 _2": -1j / 16,
                    "_1 _3 _4 _2": -1 / 16,
                    #
                    "_0 _2 _4 _3": 1j / 16,
                    "_0 _3 _4 _3": 1 / 16,
                    "_1 _2 _4 _3": 1 / 16,
                    "_1 _3 _4 _3": -1j / 16,
                    #
                    "_0 _2 _5 _2": 1j / 16,
                    "_0 _3 _5 _2": 1 / 16,
                    "_1 _2 _5 _2": 1 / 16,
                    "_1 _3 _5 _2": -1j / 16,
                    #
                    "_0 _2 _5 _3": -1 / 16,
                    "_0 _3 _5 _3": 1j / 16,
                    "_1 _2 _5 _3": 1j / 16,
                    "_1 _3 _5 _3": 1 / 16,
                },
                num_spin_orbitals=3,
            ),
        ]
        expected_ops_no_simplify = [
            MajoranaOp(
                {"_0 _2": 0.25, "_0 _3": 0.25j, "_1 _2": -0.25j, "_1 _3": 0.25}, num_spin_orbitals=2
            ),
            2
            * MajoranaOp(
                {
                    "_0 _0 _2 _2": 1 / 16,
                    "_0 _1 _2 _2": 1j / 8,
                    "_1 _1 _2 _2": 1 / 16,
                    "_0 _0 _2 _3": 1j / 8,
                    "_0 _1 _2 _3": -1 / 4,
                    "_1 _1 _2 _3": 1j / 8,
                    "_0 _0 _3 _3": 1 / 16,
                    "_0 _1 _3 _3": 1j / 8,
                    "_1 _1 _3 _3": 1 / 16,
                },
                num_spin_orbitals=2,
            ),
            3
            * MajoranaOp(
                {
                    "_0 _2 _2 _4": -1 / 16,
                    "_0 _2 _3 _4": -1j / 8,
                    "_1 _2 _2 _4": 1j / 16,
                    "_1 _2 _3 _4": -1 / 8,
                    "_0 _3 _3 _4": -1 / 16,
                    "_1 _3 _3 _4": 1j / 16,
                    "_0 _2 _2 _5": -1j / 16,
                    "_0 _2 _3 _5": 1 / 8,
                    "_1 _2 _2 _5": -1 / 16,
                    "_1 _2 _3 _5": -1j / 8,
                    "_0 _3 _3 _5": -1j / 16,
                    "_1 _3 _3 _5": -1 / 16,
                },
                num_spin_orbitals=3,
            ),
        ]
        expected_ops_no_order = [
            MajoranaOp(
                {"_0 _2": 0.25, "_0 _3": 0.25j, "_1 _2": -0.25j, "_1 _3": 0.25}, num_spin_orbitals=2
            ),
            2
            * MajoranaOp(
                {
                    "": 1 / 4,
                    "_0 _1": 1j / 8,
                    "_1 _0": -1j / 8,
                    "_2 _3": 1j / 8,
                    "_0 _1 _2 _3": -1 / 16,
                    "_1 _0 _2 _3": 1 / 16,
                    "_3 _2": -1j / 8,
                    "_0 _1 _3 _2": 1 / 16,
                    "_1 _0 _3 _2": -1 / 16,
                },
                num_spin_orbitals=2,
            ),
            3
            * MajoranaOp(
                {
                    "_0 _4": -1 / 8,
                    "_0 _5": -1j / 8,
                    "_1 _4": 1j / 8,
                    "_1 _5": -1 / 8,
                    #
                    "_0 _2 _4 _3": 1j / 16,
                    "_0 _2 _5 _3": -1 / 16,
                    "_0 _3 _4 _2": -1j / 16,
                    "_0 _3 _5 _2": 1 / 16,
                    "_1 _2 _4 _3": 1 / 16,
                    "_1 _2 _5 _3": 1j / 16,
                    "_1 _3 _4 _2": -1 / 16,
                    "_1 _3 _5 _2": -1j / 16,
                },
                num_spin_orbitals=3,
            ),
        ]
        expected_ops = [
            MajoranaOp(
                {"_0 _2": 0.25, "_0 _3": 0.25j, "_1 _2": -0.25j, "_1 _3": 0.25}, num_spin_orbitals=2
            ),
            2
            * MajoranaOp(
                {"": 1 / 4, "_0 _1": 1j / 4, "_2 _3": 1j / 4, "_0 _1 _2 _3": -1 / 4},
                num_spin_orbitals=2,
            ),
            3
            * MajoranaOp(
                {
                    "_0 _4": -1 / 8,
                    "_0 _2 _3 _4": -1j / 8,
                    "_1 _4": 1j / 8,
                    "_1 _2 _3 _4": -1 / 8,
                    "_0 _5": -1j / 8,
                    "_0 _2 _3 _5": 1 / 8,
                    "_1 _5": -1 / 8,
                    "_1 _2 _3 _5": -1j / 8,
                },
                num_spin_orbitals=3,
            ),
        ]
        with self.subTest("conversion"):
            for f_op, e_op in zip(original_ops, expected_ops):
                t_op = MajoranaOp.from_fermionic_op(f_op)
                self.assertEqual(t_op, e_op)

        with self.subTest("sum of operators"):
            f_op = original_ops[0] + original_ops[1]
            e_op = expected_ops[0] + expected_ops[1]
            t_op = MajoranaOp.from_fermionic_op(f_op)
            self.assertEqual(t_op, e_op)

        with self.subTest("composed operators"):
            f_op = original_ops[0] @ original_ops[1]
            e_op = expected_ops[0] @ expected_ops[1]
            t_op = MajoranaOp.from_fermionic_op(f_op)
            e_op_simplified = e_op.index_order().simplify()
            t_op_simplified = t_op.index_order().simplify()
            self.assertEqual(t_op_simplified, e_op_simplified)

        with self.subTest("tensored operators"):
            f_op = original_ops[0] ^ original_ops[1]
            e_op = expected_ops[0] ^ expected_ops[1]
            t_op = MajoranaOp.from_fermionic_op(f_op)
            self.assertEqual(t_op, e_op)

        with self.subTest("no simplify"):
            for f_op, e_op in zip(original_ops, expected_ops_no_simplify):
                t_op = MajoranaOp.from_fermionic_op(f_op, simplify=False)
                self.assertEqual(t_op, e_op)
        with self.subTest("no order"):
            for f_op, e_op in zip(original_ops, expected_ops_no_order):
                t_op = MajoranaOp.from_fermionic_op(f_op, order=False)
                self.assertEqual(t_op, e_op)

        with self.subTest("no simplify no order"):
            for f_op, e_op in zip(original_ops, expected_ops_no_simp_no_order):
                t_op = MajoranaOp.from_fermionic_op(f_op, simplify=False, order=False)
                self.assertEqual(t_op, e_op)


if __name__ == "__main__":
    unittest.main()
