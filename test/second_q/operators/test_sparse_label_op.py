# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for SparseLabelOp"""

from __future__ import annotations

from typing import Collection, Iterator, Mapping

import unittest
from test import QiskitNatureTestCase

import numpy as np

from qiskit.circuit import Parameter

from qiskit_nature.second_q.operators import PolynomialTensor, SparseLabelOp


a = Parameter("a")
b = Parameter("b")

op1 = {
    "+_0 -_1": 0.0,
    "+_0 -_2": 1.0,
}

op2 = {
    "+_0 -_1": 0.5,
    "+_0 -_2": 1.0,
}

op3 = {
    "+_0 -_1": 0.5,
    "+_0 -_3": 3.0,
}

opComplex = {
    "+_0 -_1": 0.5 + 1j,
    "+_0 -_2": 1.0,
}

opParameter = {
    "+_0 -_1": a,
    "+_0 -_2": b,
}


class DummySparseLabelOp(SparseLabelOp):
    """Dummy SparseLabelOp for testing purposes"""

    @property
    def register_length(self) -> int | None:
        return None

    # pylint: disable=unused-argument
    def _new_instance(
        self, data: Mapping[str, complex], *, other: SparseLabelOp | None = None
    ) -> SparseLabelOp:
        return self.__class__(data, copy=False)

    def _validate_keys(self, keys: Collection[str]) -> None:
        pass

    @classmethod
    def _validate_polynomial_tensor_key(cls, keys: Collection[str]) -> None:
        pass

    @classmethod
    def from_polynomial_tensor(cls, tensor: PolynomialTensor) -> SparseLabelOp:
        pass

    def terms(self) -> Iterator[tuple[list[tuple[str, int]], complex]]:
        pass

    def transpose(self) -> SparseLabelOp:
        return self

    def compose(self, other, qargs=None, front=False) -> SparseLabelOp:
        return self

    def tensor(self, other) -> SparseLabelOp:
        return self

    def expand(self, other) -> SparseLabelOp:
        return self

    # pylint: disable=unused-argument
    def simplify(self, atol: float | None = None) -> SparseLabelOp:
        return self


class TestSparseLabelOp(QiskitNatureTestCase):
    """SparseLabelOp tests."""

    def test_add(self):
        """Test add method"""
        with self.subTest("real + real"):
            test_op = DummySparseLabelOp(op1) + DummySparseLabelOp(op2)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.5,
                    "+_0 -_2": 2.0,
                },
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("complex + real"):
            test_op = DummySparseLabelOp(op2) + DummySparseLabelOp(opComplex)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 1.0 + 1j,
                    "+_0 -_2": 2.0,
                },
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("complex + complex"):
            test_op = DummySparseLabelOp(opComplex) + DummySparseLabelOp(opComplex)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 1.0 + 2j,
                    "+_0 -_2": 2.0,
                },
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("complex + parameter"):
            test_op = DummySparseLabelOp(opComplex) + DummySparseLabelOp(opParameter)
            target_op = DummySparseLabelOp({"+_0 -_1": 0.5 + 1j + a, "+_0 -_2": 1.0 + b})

            self.assertEqual(test_op, target_op)

        with self.subTest("new key"):
            test_op = DummySparseLabelOp(op1) + DummySparseLabelOp(op3)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.5,
                    "+_0 -_2": 1.0,
                    "+_0 -_3": 3.0,
                },
            )

            self.assertEqual(test_op, target_op)

    def test_mul(self):
        """Test scalar multiplication method"""
        with self.subTest("real * real"):
            test_op = DummySparseLabelOp(op1) * 2
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.0,
                    "+_0 -_2": 2.0,
                },
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("complex * real"):
            test_op = DummySparseLabelOp(opComplex) * 2
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 1.0 + 2j,
                    "+_0 -_2": 2.0,
                },
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("real * complex"):
            test_op = DummySparseLabelOp(op2) * (0.5 + 1j)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.25 + 0.5j,
                    "+_0 -_2": 0.5 + 1j,
                },
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("complex * complex"):
            test_op = DummySparseLabelOp(opComplex) * (0.5 + 1j)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": -0.75 + 1j,
                    "+_0 -_2": 0.5 + 1j,
                },
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("parameter * complex"):
            test_op = DummySparseLabelOp(opParameter) * (0.5 + 1j)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": a * (0.5 + 1j),
                    "+_0 -_2": b * (0.5 + 1j),
                },
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("complex * parameter"):
            test_op = DummySparseLabelOp(opComplex) * (a + b)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": (0.5 + 1j) * (a + b),
                    "+_0 -_2": (a + b),
                },
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("parameter * parameter"):
            test_op = DummySparseLabelOp(opParameter) * (a + b)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": a * (a + b),
                    "+_0 -_2": b * (a + b),
                },
            )

            self.assertEqual(test_op, target_op)

        with self.subTest("raises TypeError"):
            with self.assertRaises(TypeError):
                _ = DummySparseLabelOp(op1) * "something"

        # regression test against https://github.com/Qiskit/qiskit-nature/issues/953
        with self.subTest("numpy types"):
            test_op = np.double(2) * DummySparseLabelOp(op1)
            target_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.0,
                    "+_0 -_2": 2.0,
                },
            )

            self.assertEqual(test_op, target_op)

    def test_adjoint(self):
        """Test adjoint method"""
        with self.subTest("complex"):
            test_op = DummySparseLabelOp(opComplex).adjoint()
            target_op = DummySparseLabelOp({"+_0 -_1": 0.5 - 1j, "+_0 -_2": 1.0})
            self.assertEqual(test_op, target_op)

        with self.subTest("parameter"):
            test_op = DummySparseLabelOp(opParameter).adjoint()
            target_op = DummySparseLabelOp({"+_0 -_1": a.conjugate(), "+_0 -_2": b.conjugate()})
            self.assertEqual(test_op, target_op)

    def test_conjugate(self):
        """Test conjugate method"""
        with self.subTest("complex"):
            test_op = DummySparseLabelOp(opComplex).conjugate()
            target_op = DummySparseLabelOp({"+_0 -_1": 0.5 - 1j, "+_0 -_2": 1.0})
            self.assertEqual(test_op, target_op)

        with self.subTest("parameter"):
            test_op = DummySparseLabelOp(opParameter).conjugate()
            target_op = DummySparseLabelOp({"+_0 -_1": a.conjugate(), "+_0 -_2": b.conjugate()})
            self.assertEqual(test_op, target_op)

    def test_eq(self):
        """test __eq__ method"""
        with self.subTest("equal"):
            test_op = DummySparseLabelOp(op1) == DummySparseLabelOp(op1)
            self.assertTrue(test_op)

        with self.subTest("not equal - keys"):
            test_op = DummySparseLabelOp(op1) == DummySparseLabelOp(
                {
                    "+_0 -_1": 0.0,
                    "+_0 -_3": 1.0,
                },
            )
            self.assertFalse(test_op)

        with self.subTest("not equal - values"):
            test_op = DummySparseLabelOp(op1) == DummySparseLabelOp(op2)
            self.assertFalse(test_op)

        with self.subTest("not equal - tolerance"):
            test_op = DummySparseLabelOp(op1) == DummySparseLabelOp(
                {
                    "+_0 -_1": 0.000000001,
                    "+_0 -_2": 1.0,
                },
            )

            self.assertFalse(test_op)

    def test_equiv(self):
        """test equiv method"""
        with self.subTest("not equivalent - tolerances"):
            test_op = DummySparseLabelOp(op1).equiv(
                DummySparseLabelOp(
                    {
                        "+_0 -_1": 0.000001,
                        "+_0 -_2": 1.0,
                    },
                )
            )

            self.assertFalse(test_op)

        with self.subTest("not equivalent - keys"):
            test_op = DummySparseLabelOp(op1).equiv(
                DummySparseLabelOp(
                    {
                        "+_0 -_1": 0.0,
                        "+_0 -_3": 1.0,
                    },
                )
            )

            self.assertFalse(test_op)

        with self.subTest("equivalent"):
            test_op = DummySparseLabelOp(op1).equiv(
                DummySparseLabelOp(
                    {
                        "+_0 -_1": 0.000000001,
                        "+_0 -_2": 1.0,
                    },
                )
            )

            self.assertTrue(test_op)

        with self.subTest("parameters"):
            test_op = DummySparseLabelOp(opParameter)
            with self.assertRaisesRegex(ValueError, "parameter"):
                _ = test_op.equiv(DummySparseLabelOp(opParameter))
            test_op = DummySparseLabelOp(opComplex)
            with self.assertRaisesRegex(ValueError, "parameter"):
                _ = test_op.equiv(DummySparseLabelOp(opParameter))

    def test_iter(self):
        """test __iter__ method"""
        test_op = iter(DummySparseLabelOp(op1))

        self.assertEqual(next(test_op), "+_0 -_1")
        self.assertEqual(next(test_op), "+_0 -_2")

    def test_get_item(self):
        """test __getitem__ method"""
        test_op = DummySparseLabelOp(op1)
        self.assertEqual(test_op["+_0 -_1"], 0.0)

    def test_len(self):
        """test __len__ method"""
        test_op = DummySparseLabelOp(op1)
        self.assertEqual(len(test_op), 2)

    def test_copy(self):
        """test copy bool"""
        data = {
            "+_0 -_1": 0.0,
            "+_0 -_3": 1.0,
        }
        test_op = DummySparseLabelOp(data, copy=True)
        data["+_0 -_1"] = 0.2
        self.assertEqual(test_op._data["+_0 -_1"], 0.0)

    def test_zero(self):
        """test zero class initializer"""
        test_op = DummySparseLabelOp.zero()
        self.assertEqual(test_op._data, {})

    def test_one(self):
        """test one class initializer"""
        test_op = DummySparseLabelOp.one()
        self.assertEqual(test_op._data, {"": 1.0})

    def test_induced_norm(self):
        """Test induced norm."""
        op = DummySparseLabelOp({"+_0 -_1": 3.0, "+_0 -_2": -4j})
        self.assertAlmostEqual(op.induced_norm(), 7.0)
        self.assertAlmostEqual(op.induced_norm(2), 5.0)

        test_op = DummySparseLabelOp(opParameter)
        with self.assertRaisesRegex(ValueError, "parameter"):
            _ = test_op.induced_norm()

    def test_chop(self):
        """Test chop."""
        op = DummySparseLabelOp({"+_0 -_1": 1 + 1e-12j, "+_0 -_2": a})
        self.assertEqual(op.chop(), DummySparseLabelOp({"+_0 -_1": 1, "+_0 -_2": a}))

        op = DummySparseLabelOp({"+_0 -_1": 1e-12 + 1j, "+_0 -_2": a})
        self.assertEqual(op.chop(), DummySparseLabelOp({"+_0 -_1": 1j, "+_0 -_2": a}))

        self.assertEqual((op - op).chop(), DummySparseLabelOp.zero())

    def test_is_parameterized(self):
        """Test is_parameterized."""
        self.assertTrue(DummySparseLabelOp(opParameter).is_parameterized())
        self.assertFalse(DummySparseLabelOp(op1).is_parameterized())

    def test_assign_parameters(self):
        """Test assign_parameters."""
        op = DummySparseLabelOp({"+_0 -_1": a, "+_0 -_2": b})
        assigned_op = op.assign_parameters({a: 1.0})
        self.assertEqual(assigned_op, DummySparseLabelOp({"+_0 -_1": 1.0, "+_0 -_2": b}))
        self.assertEqual(op, DummySparseLabelOp({"+_0 -_1": a, "+_0 -_2": b}))

    def test_round(self):
        """test round function"""
        with self.subTest("round just real part"):
            data = {
                "+_0 -_1": 0.7 + 3j,
                "+_0 -_3": 1.1 + 4j,
            }
            test_op = DummySparseLabelOp(data).round()
            self.assertEqual(
                test_op._data,
                {
                    "+_0 -_1": 1.0 + 3j,
                    "+_0 -_3": 1.0 + 4j,
                },
            )

        with self.subTest("round just imag part"):
            data = {
                "+_0 -_1": 1.0 + 0.9j,
                "+_0 -_3": 1.0 + 0.2j,
            }
            test_op = DummySparseLabelOp(data).round()
            self.assertEqual(
                test_op._data,
                {
                    "+_0 -_1": 1.0 + 1j,
                    "+_0 -_3": 1.0 + 0j,
                },
            )

        with self.subTest("round real and imag part"):
            data = {
                "+_0 -_1": 0.8 + 0.3j,
                "+_0 -_3": 1.2 + 0.8j,
            }
            test_op = DummySparseLabelOp(data).round()
            self.assertEqual(
                test_op._data,
                {
                    "+_0 -_1": 1 + 0j,
                    "+_0 -_3": 1.0 + 1j,
                },
            )

        with self.subTest("round real and imag part to 3dp"):
            data = {
                "+_0 -_1": 0.8762 + 0.3789j,
                "+_0 -_3": 1.2458 + 0.8652j,
            }
            test_op = DummySparseLabelOp(data).round(3)
            self.assertEqual(
                test_op._data,
                {
                    "+_0 -_1": 0.876 + 0.379j,
                    "+_0 -_3": 1.246 + 0.865j,
                },
            )

        with self.subTest("round just real part to 3dp"):
            data = {
                "+_0 -_1": 0.8762 + 0.370j,
                "+_0 -_3": 1.2458 + 0.860j,
            }
            test_op = DummySparseLabelOp(data).round(3)
            self.assertEqual(
                test_op._data,
                {
                    "+_0 -_1": 0.876 + 0.370j,
                    "+_0 -_3": 1.246 + 0.860j,
                },
            )

        with self.subTest("round just imag part to 3dp"):
            data = {
                "+_0 -_1": 0.8760 + 0.3789j,
                "+_0 -_3": 1.245 + 0.8652j,
            }
            test_op = DummySparseLabelOp(data).round(3)
            self.assertEqual(
                test_op._data,
                {
                    "+_0 -_1": 0.8760 + 0.379j,
                    "+_0 -_3": 1.245 + 0.865j,
                },
            )

    def test_is_zero(self):
        """test if coefficients are all zero"""
        with self.subTest("operator length is zero"):
            test_op = DummySparseLabelOp({})
            self.assertTrue(test_op.is_zero())

        with self.subTest("coefficients are all zero"):
            test_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.0,
                    "+_0 -_3": 0.0,
                }
            )
            self.assertTrue(test_op.is_zero())

        with self.subTest("coefficients are all zero with tol"):
            test_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.05,
                    "+_0 -_3": 0.0,
                }
            )
            self.assertTrue(test_op.is_zero(tol=0.1))

        with self.subTest("coefficients are all zero with smaller val"):
            test_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.0,
                    "+_0 -_3": 1e-18,
                }
            )
            self.assertTrue(test_op.is_zero())

        with self.subTest("coefficients not all zero"):
            test_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.0,
                    "+_0 -_3": 0.1,
                }
            )
            self.assertFalse(test_op.is_zero())

        with self.subTest("coefficients not all zero with tol"):
            test_op = DummySparseLabelOp(
                {
                    "+_0 -_1": 0.05,
                    "+_0 -_3": 0.0,
                }
            )
            self.assertFalse(test_op.is_zero(tol=0.001))

    def test_parameters(self):
        """Test parameters."""
        op = DummySparseLabelOp({"+_0 -_1": a, "+_0 -_2": b})
        self.assertEqual(op.parameters(), [a, b])


if __name__ == "__main__":
    unittest.main()
