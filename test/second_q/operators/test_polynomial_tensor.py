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

"""Test for PolynomialTensor class"""

from __future__ import annotations

import unittest
from test import QiskitNatureTestCase

import numpy as np
import sparse as sp
from ddt import ddt, idata

from qiskit.test import slow_test
from qiskit_nature.second_q.operators import PolynomialTensor


@ddt
class TestPolynomialTensor(QiskitNatureTestCase):
    """Tests for PolynomialTensor class"""

    def setUp(self) -> None:
        super().setUp()

        self.og_poly = {
            "": 1.0,
            "+": self.build_matrix(4, 1),
            "+-": self.build_matrix(4, 2),
            "++--": self.build_matrix(4, 4),
        }

        self.og_transposed = {
            "": 1.0,
            "+": self.build_matrix(4, 1).transpose(),
            "+-": self.build_matrix(4, 2).transpose(),
            "++--": self.build_matrix(4, 4).transpose(),
        }

        self.kronecker = {
            "": 1.0,
            "+": np.array([1, 1]),
            "+-": np.array([[1, 0], [0, 1]]),
            "++--": np.fromiter(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=float,
            ).reshape((2, 2, 2, 2)),
        }

        self.expected_tensor_poly = {
            "": 1.0,
            "+": np.concatenate([self.build_matrix(4, 1), self.build_matrix(4, 1)]),
            "+-": np.kron(self.kronecker["+-"], self.build_matrix(4, 2)),
            "++--": np.kron(self.kronecker["++--"], self.build_matrix(4, 4)),
        }

        self.sparse_1 = {
            "": 1.0,
            "+": sp.as_coo({(0,): 1, (2,): 2}, shape=(4,)),
            "+-": sp.as_coo({(0, 0): 1, (1, 0): 2}, shape=(4, 4)),
            "++--": sp.as_coo({(0, 0, 0, 1): 1, (1, 0, 2, 1): 2}, shape=(4, 4, 4, 4)),
        }

        self.sparse_1_transposed = {
            "": 1.0,
            "+": sp.as_coo({(0,): 1, (2,): 2}, shape=(4,)),
            "+-": sp.as_coo({(0, 0): 1, (0, 1): 2}, shape=(4, 4)),
            "++--": sp.as_coo({(1, 0, 0, 0): 1, (1, 2, 0, 1): 2}, shape=(4, 4, 4, 4)),
        }

        self.sparse_2 = {
            "": 2.0,
            "+": sp.as_coo({(1,): 1}, shape=(4,)),
            "+-": sp.as_coo({(0, 1): 1, (1, 0): 2}, shape=(4, 4)),
            "++--": sp.as_coo({(0, 1, 0, 1): 1, (1, 0, 2, 1): -2}, shape=(4, 4, 4, 4)),
        }

        self.sparse_kronecker = {
            "": 1.0,
            "+": sp.as_coo({(0,): 1, (1,): 1}, shape=(2,)),
            "+-": sp.as_coo({(0, 0): 1, (1, 1): 1}, shape=(2, 2)),
            "++--": sp.as_coo(
                {(0, 0, 0, 0): 1, (1, 0, 0, 1): 1, (0, 1, 1, 0): 1, (1, 1, 1, 1): 1},
                shape=(2, 2, 2, 2),
            ),
        }

        self.expected_sparse_tensor_poly = {
            "": 1.0,
            "+": sp.as_coo({(0,): 1, (2,): 2, (4,): 1, (6,): 2}, shape=(8,)),
            "+-": sp.as_coo({(0, 0): 1, (1, 0): 2, (4, 4): 1, (5, 4): 2}, shape=(8, 8)),
            "++--": sp.as_coo(
                {
                    (0, 0, 0, 1): 1,
                    (0, 4, 4, 1): 1,
                    (1, 0, 2, 1): 2,
                    (1, 4, 6, 1): 2,
                    (4, 0, 0, 5): 1,
                    (4, 4, 4, 5): 1,
                    (5, 0, 2, 5): 2,
                    (5, 4, 6, 5): 2,
                },
                shape=(8, 8, 8, 8),
            ),
        }

    @staticmethod
    def build_matrix(dim_size, num_dim, val=1):
        """Build dictionary value matrix"""
        return (np.arange(1, dim_size**num_dim + 1) * val).reshape((dim_size,) * num_dim)

    def test_init(self):
        """Test for errors in constructor for PolynomialTensor"""
        with self.subTest("normal dense"):
            _ = PolynomialTensor(self.og_poly)

        with self.subTest("normal sparse"):
            _ = PolynomialTensor(self.sparse_1)

        with self.assertRaisesRegex(
            ValueError,
            r"Data key .* of length \d does not match data value matrix of dimensions \(\d+, *\)",
        ):
            _ = PolynomialTensor(
                {
                    "++": self.build_matrix(4, 1),
                }
            )

        with self.assertRaisesRegex(
            ValueError, r"For key (.*): dimensions of value matrix are not identical \(\d+, .*\)"
        ):
            _ = PolynomialTensor(
                {
                    "+-": self.build_matrix(4, 2),
                    "++--": np.arange(1, 13).reshape(1, 2, 3, 2),
                }
            )

    def test_is_empty(self):
        """Test PolynomialTensor.is_empty"""
        with self.subTest("empty"):
            self.assertTrue(PolynomialTensor.empty().is_empty())

        with self.subTest("non-empty"):
            self.assertFalse(PolynomialTensor({"": 1.0}).is_empty())

    def test_contains_sparse(self):
        """Test PolynomialTensor.contains_sparse"""
        with self.subTest("sparse"):
            self.assertTrue(PolynomialTensor({"+": sp.as_coo({(0,): 1})}).contains_sparse())

        with self.subTest("non-sparse"):
            self.assertFalse(PolynomialTensor({"+": np.array([1])}).contains_sparse())

    def test_get_item(self):
        """Test for getting value matrices corresponding to keys in PolynomialTensor"""
        og_poly_tensor = PolynomialTensor(self.og_poly)
        for key, value in self.og_poly.items():
            np.testing.assert_array_equal(value, og_poly_tensor[key])

    def test_len(self):
        """Test for the length of PolynomialTensor"""
        length = len(
            PolynomialTensor(
                {
                    "": 1.0,
                    "+": self.build_matrix(2, 1),
                    "+-": self.build_matrix(2, 2),
                    "++--": self.build_matrix(2, 4),
                }
            )
        )

        exp_len = 4
        self.assertEqual(exp_len, length)

    def test_iter(self):
        """Test for the iterator of PolynomialTensor"""
        og_poly_tensor = PolynomialTensor(self.og_poly)
        exp_iter = [key for key, _ in self.og_poly.items()]
        self.assertEqual(exp_iter, list(iter(og_poly_tensor)))

    def test_todense(self):
        """Test PolynomialTensor.todense"""
        dense_tensor = PolynomialTensor(self.sparse_1).todense()
        two_body = np.zeros((4, 4, 4, 4))
        two_body[0, 0, 0, 1] = 1
        two_body[1, 0, 2, 1] = 2
        expected = {
            "": 1.0,
            "+": np.array([1, 0, 2, 0]),
            "+-": np.array([[1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            "++--": two_body,
        }
        self.assertEqual(dense_tensor, PolynomialTensor(expected))

    def tosparse(self):
        """Test PolynomialTensor.tosparse"""
        sparse_tensor = PolynomialTensor(self.og_poly).tosparse()
        expected = {
            "": 1.0,
            "+": sp.as_coo(np.arange(1, 5)),
            "+-": sp.as_coo(np.arange(1, 17).reshape((4, 4))),
            "++--": sp.as_coo(np.arange(1, 257).reshape((4, 4, 4, 4))),
        }
        self.assertEqual(sparse_tensor, PolynomialTensor(expected))

    @idata(np.linspace(0, 3, 5))
    def test_mul(self, other):
        """Test for scalar multiplication"""

        with self.subTest("dense"):
            expected = {
                "": 1.0 * other,
                "+": self.build_matrix(4, 1, other),
                "+-": self.build_matrix(4, 2, other),
                "++--": self.build_matrix(4, 4, other),
            }

            result = PolynomialTensor(self.og_poly) * other
            self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("sparse"):
            expected = {
                "": other,
                "+": other * sp.as_coo({(0,): 1, (2,): 2}, shape=(4,)),
                "+-": other * sp.as_coo({(0, 0): 1, (1, 0): 2}, shape=(4, 4)),
                "++--": other * sp.as_coo({(0, 0, 0, 1): 1, (1, 0, 2, 1): 2}, shape=(4, 4, 4, 4)),
            }
            result = PolynomialTensor(self.sparse_1) * other
            self.assertEqual(result, PolynomialTensor(expected))

        with self.assertRaises(TypeError):
            _ = PolynomialTensor(self.og_poly) * PolynomialTensor(self.og_poly)

    def test_add(self):
        """Test for addition of PolynomialTensor"""

        with self.subTest("dense + dense"):
            result = PolynomialTensor(self.og_poly) + PolynomialTensor(self.og_poly)
            expected = {
                "": 2.0,
                "+": np.add(self.build_matrix(4, 1), self.build_matrix(4, 1)),
                "+-": np.add(self.build_matrix(4, 2), self.build_matrix(4, 2)),
                "++--": np.add(self.build_matrix(4, 4), self.build_matrix(4, 4)),
            }
            self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("sparse + dense"):
            result = PolynomialTensor(self.sparse_1) + PolynomialTensor(self.og_poly)
            expected = {
                "": 2.0,
                "+": self.build_matrix(4, 1),
                "+-": self.build_matrix(4, 2),
                "++--": self.build_matrix(4, 4),
            }
            expected["+"][0] += 1
            expected["+"][2] += 2
            expected["+-"][0, 0] += 1
            expected["+-"][1, 0] += 2
            expected["++--"][0, 0, 0, 1] += 1
            expected["++--"][1, 0, 2, 1] += 2
            self.assertEqual(result, PolynomialTensor(expected))
            self.assertIsInstance(result["+"], np.ndarray)
            self.assertIsInstance(result["+-"], np.ndarray)
            self.assertIsInstance(result["++--"], np.ndarray)

        with self.subTest("sparse + sparse"):
            result = PolynomialTensor(self.sparse_1) + PolynomialTensor(self.sparse_2)
            expected = {
                "": 3.0,
                "+": sp.as_coo({(0,): 1, (1,): 1, (2,): 2}, shape=(4,)),
                "+-": sp.as_coo({(0, 0): 1, (0, 1): 1, (1, 0): 4}, shape=(4, 4)),
                "++--": sp.as_coo({(0, 0, 0, 1): 1, (0, 1, 0, 1): 1}, shape=(4, 4, 4, 4)),
            }
            self.assertEqual(result, PolynomialTensor(expected))
            self.assertIsInstance(result["+"], sp.COO)
            self.assertIsInstance(result["+-"], sp.COO)
            self.assertIsInstance(result["++--"], sp.COO)

        with self.assertRaisesRegex(
            TypeError, "Incorrect argument type: other should be PolynomialTensor"
        ):
            _ = PolynomialTensor(self.og_poly) + 5

    def test_conjugate(self):
        """Test for conjugate of PolynomialTensor"""

        with self.subTest("dense"):
            result = PolynomialTensor(
                {
                    "": 1 + 1j,
                    "+": self.build_matrix(4, 1, 1j),
                    "+-": self.build_matrix(4, 2, 1j),
                    "++--": self.build_matrix(4, 4, 1j),
                }
            ).conjugate()
            expected = PolynomialTensor(
                {
                    "": 1 - 1j,
                    "+": self.build_matrix(4, 1, -1j),
                    "+-": self.build_matrix(4, 2, -1j),
                    "++--": self.build_matrix(4, 4, -1j),
                }
            )
            self.assertEqual(result, expected)

        with self.subTest("sparse"):
            result = PolynomialTensor(
                {
                    "": 1 + 1j,
                    "+": sp.as_coo({(0,): 1j, (2,): 2j}, shape=(4,)),
                    "+-": sp.as_coo({(0, 0): 1j, (1, 0): 2j}, shape=(4, 4)),
                    "++--": sp.as_coo({(0, 0, 0, 1): 1j, (1, 0, 2, 1): 2j}, shape=(4, 4, 4, 4)),
                }
            ).conjugate()
            expected = PolynomialTensor(
                {
                    "": 1 - 1j,
                    "+": sp.as_coo({(0,): -1j, (2,): -2j}, shape=(4,)),
                    "+-": sp.as_coo({(0, 0): -1j, (1, 0): -2j}, shape=(4, 4)),
                    "++--": sp.as_coo({(0, 0, 0, 1): -1j, (1, 0, 2, 1): -2j}, shape=(4, 4, 4, 4)),
                }
            )
            self.assertEqual(result, expected)

    def test_transpose(self):
        """Test for transpose of PolynomialTensor"""

        with self.subTest("dense"):
            result = PolynomialTensor(self.og_poly).transpose()
            self.assertEqual(result, PolynomialTensor(self.og_transposed))

        with self.subTest("sparse"):
            result = PolynomialTensor(self.sparse_1).transpose()
            self.assertEqual(result, PolynomialTensor(self.sparse_1_transposed))

    def test_compose(self):
        """Test composition of PolynomialTensor"""

        with self.subTest("dense with dense"):
            pt_a = PolynomialTensor(self.og_poly)
            pt_b = PolynomialTensor(self.og_transposed)

            with self.subTest("compose(front=False)"):
                result = pt_a.compose(pt_b)
                expected = {
                    "": 1.0,
                    "+": np.multiply(self.build_matrix(4, 1).transpose(), self.build_matrix(4, 1)),
                    "+-": np.matmul(self.build_matrix(4, 2).transpose(), self.build_matrix(4, 2)),
                    "++--": np.matmul(self.build_matrix(4, 4).transpose(), self.build_matrix(4, 4)),
                }
                self.assertEqual(result, PolynomialTensor(expected))

            with self.subTest("compose(front=True)"):
                result = pt_a.compose(pt_b, front=True)
                expected = {
                    "": 1.0,
                    "+": np.multiply(self.build_matrix(4, 1), self.build_matrix(4, 1).transpose()),
                    "+-": np.matmul(self.build_matrix(4, 2), self.build_matrix(4, 2).transpose()),
                    "++--": np.matmul(self.build_matrix(4, 4), self.build_matrix(4, 4).transpose()),
                }
                self.assertEqual(result, PolynomialTensor(expected))

    @slow_test
    def test_compose_sparse(self):
        """Test composition of sparse PolynomialTensor"""

        with self.subTest("sparse with dense"):
            pt_a = PolynomialTensor(self.sparse_1)
            pt_b = PolynomialTensor(self.og_poly)

            with self.subTest("compose(front=False)"):
                result = pt_a.compose(pt_b)
                expected = {
                    "": 1.0,
                    "+": sp.as_coo({(0,): 1, (2,): 6}, shape=(4,)),
                    "+-": sp.as_coo(
                        {(0, 0): 5, (1, 0): 17, (2, 0): 29, (3, 0): 41}, shape=(4, 4)
                    ).todense(),
                    "++--": sp.as_coo(
                        {
                            (0, 0, 0, 1): 1,
                            (0, 0, 1, 1): 5,
                            (0, 0, 2, 1): 9,
                            (0, 0, 3, 1): 13,
                            (1, 0, 0, 1): 134,
                            (1, 0, 1, 1): 142,
                            (1, 0, 2, 1): 150,
                            (1, 0, 3, 1): 158,
                        },
                        shape=(4, 4, 4, 4),
                    ).todense(),
                }
                self.assertEqual(result, PolynomialTensor(expected))

            with self.subTest("compose(front=True)"):
                result = pt_a.compose(pt_b, front=True)
                expected = {
                    "": 1.0,
                    "+": sp.as_coo({(0,): 1, (2,): 6}, shape=(4,)),
                    "+-": sp.as_coo(
                        {
                            (0, 0): 1,
                            (0, 1): 2,
                            (0, 2): 3,
                            (0, 3): 4,
                            (1, 0): 2,
                            (1, 1): 4,
                            (1, 2): 6,
                            (1, 3): 8,
                        },
                        shape=(4, 4),
                    ).todense(),
                    "++--": sp.as_coo(
                        {
                            (0, 0, 0, 0): 5,
                            (0, 0, 0, 1): 6,
                            (0, 0, 0, 2): 7,
                            (0, 0, 0, 3): 8,
                            (1, 0, 2, 0): 138,
                            (1, 0, 2, 1): 140,
                            (1, 0, 2, 2): 142,
                            (1, 0, 2, 3): 144,
                        },
                        shape=(4, 4, 4, 4),
                    ).todense(),
                }
                self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("sparse with sparse"):
            pt_a = PolynomialTensor(self.sparse_1)
            pt_b = PolynomialTensor(self.sparse_1_transposed)

            with self.subTest("compose(front=False)"):
                result = pt_a.compose(pt_b)
                expected = {
                    "": 1.0,
                    "+": sp.as_coo({(0,): 1, (2,): 4}, shape=(4,)),
                    "+-": sp.as_coo({(0, 0): 5}, shape=(4, 4)),
                    "++--": sp.as_coo({}, shape=(4, 4, 4, 4)),
                }
                self.assertEqual(result, PolynomialTensor(expected))

            with self.subTest("compose(front=True)"):
                result = pt_a.compose(pt_b, front=True)
                expected = {
                    "": 1.0,
                    "+": sp.as_coo({(0,): 1, (2,): 4}, shape=(4,)),
                    "+-": sp.as_coo({(0, 0): 1, (0, 1): 2, (1, 0): 2, (1, 1): 4}, shape=(4, 4)),
                    "++--": sp.as_coo({}, shape=(4, 4, 4, 4)),
                }
                self.assertEqual(result, PolynomialTensor(expected))

    def test_tensor(self):
        """Test tensoring of PolynomialTensor"""

        with self.subTest("dense with dense"):
            p_t = PolynomialTensor(self.og_poly)
            result = PolynomialTensor(self.kronecker).tensor(p_t)
            self.assertEqual(result, PolynomialTensor(self.expected_tensor_poly))

        with self.subTest("sparse with dense"):
            p_t = PolynomialTensor(self.og_poly)
            result = PolynomialTensor(self.sparse_kronecker).tensor(p_t)
            self.assertEqual(result, PolynomialTensor(self.expected_tensor_poly))

        with self.subTest("sparse with sparse"):
            p_t = PolynomialTensor(self.sparse_1)
            result = PolynomialTensor(self.sparse_kronecker).tensor(p_t)
            self.assertEqual(result, PolynomialTensor(self.expected_sparse_tensor_poly))

        with self.subTest("dense with sparse"):
            p_t = PolynomialTensor(self.sparse_1)
            result = PolynomialTensor(self.kronecker).tensor(p_t)
            self.assertEqual(result, PolynomialTensor(self.expected_sparse_tensor_poly))

    def test_expand(self):
        """Test expanding of PolynomialTensor"""

        with self.subTest("dense with dense"):
            p_t = PolynomialTensor(self.og_poly)
            result = p_t.expand(PolynomialTensor(self.kronecker))
            self.assertEqual(result, PolynomialTensor(self.expected_tensor_poly))

        with self.subTest("dense with sparse"):
            p_t = PolynomialTensor(self.og_poly)
            result = p_t.expand(PolynomialTensor(self.sparse_kronecker))
            self.assertEqual(result, PolynomialTensor(self.expected_tensor_poly))

        with self.subTest("sparse with sparse"):
            p_t = PolynomialTensor(self.sparse_1)
            result = p_t.expand(PolynomialTensor(self.sparse_kronecker))
            self.assertEqual(result, PolynomialTensor(self.expected_sparse_tensor_poly))

        with self.subTest("sparse with dense"):
            p_t = PolynomialTensor(self.sparse_1)
            result = p_t.expand(PolynomialTensor(self.kronecker))
            self.assertEqual(result, PolynomialTensor(self.expected_sparse_tensor_poly))

    def test_einsum(self):
        """Test PolynomialTensor.einsum"""

        with self.subTest("all dense"):
            one_body = np.random.random((2, 2))
            two_body = np.random.random((2, 2, 2, 2))
            tensor = PolynomialTensor({"+-": one_body, "++--": two_body})
            coeffs = np.random.random((2, 2))
            coeffs_pt = PolynomialTensor({"+-": coeffs})

            result = PolynomialTensor.einsum(
                {
                    "jk,ji,kl->il": ("+-", "+-", "+-", "+-"),
                    "pqrs,pi,qj,rk,sl->ijkl": ("++--", "+-", "+-", "+-", "+-", "++--"),
                },
                tensor,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
            )

            expected = PolynomialTensor(
                {
                    "+-": np.dot(np.dot(coeffs.T, one_body), coeffs),
                    "++--": np.einsum(
                        "pqrs,pi,qj,rk,sl->ijkl",
                        two_body,
                        coeffs,
                        coeffs,
                        coeffs,
                        coeffs,
                        optimize=True,
                    ),
                }
            )

            self.assertTrue(result.equiv(expected))

        with self.subTest("all sparse"):
            one_body = sp.random((2, 2), density=0.5)
            two_body = sp.random((2, 2, 2, 2), density=0.5)
            tensor = PolynomialTensor({"+-": one_body, "++--": two_body})
            coeffs = sp.random((2, 2), density=0.5)
            coeffs_pt = PolynomialTensor({"+-": coeffs})

            result = PolynomialTensor.einsum(
                {
                    "jk,ji,kl->il": ("+-", "+-", "+-", "+-"),
                    "pqrs,pi,qj,rk,sl->ijkl": ("++--", "+-", "+-", "+-", "+-", "++--"),
                },
                tensor,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
            )

            dense_one_body = one_body.todense()
            dense_two_body = two_body.todense()
            dense_coeffs = coeffs.todense()
            expected = PolynomialTensor(
                {
                    "+-": np.dot(np.dot(dense_coeffs.T, dense_one_body), dense_coeffs),
                    "++--": np.einsum(
                        "pqrs,pi,qj,rk,sl->ijkl",
                        dense_two_body,
                        dense_coeffs,
                        dense_coeffs,
                        dense_coeffs,
                        dense_coeffs,
                        optimize=True,
                    ),
                }
            )

            self.assertTrue(result.equiv(expected))

        with self.subTest("mixed"):
            one_body = sp.random((2, 2), density=0.5)
            two_body = sp.random((2, 2, 2, 2), density=0.5)
            tensor = PolynomialTensor({"+-": one_body, "++--": two_body})
            coeffs = np.random.random((2, 2))
            coeffs_pt = PolynomialTensor({"+-": coeffs})

            result = PolynomialTensor.einsum(
                {
                    "jk,ji,kl->il": ("+-", "+-", "+-", "+-"),
                    "pqrs,pi,qj,rk,sl->ijkl": ("++--", "+-", "+-", "+-", "+-", "++--"),
                },
                tensor,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
                coeffs_pt,
            )

            dense_one_body = one_body.todense()
            dense_two_body = two_body.todense()
            expected = PolynomialTensor(
                {
                    "+-": np.dot(np.dot(coeffs.T, dense_one_body), coeffs),
                    "++--": np.einsum(
                        "pqrs,pi,qj,rk,sl->ijkl",
                        dense_two_body,
                        coeffs,
                        coeffs,
                        coeffs,
                        coeffs,
                        optimize=True,
                    ),
                }
            )

            self.assertTrue(result.equiv(expected))


if __name__ == "__main__":
    unittest.main()
