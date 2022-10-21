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
from ddt import ddt, idata

from qiskit_nature.second_q.operators import PolynomialTensor
import qiskit_nature.optionals as _optionals


@ddt
class TestPolynomialTensor(QiskitNatureTestCase):
    """Tests for PolynomialTensor class"""

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def setUp(self) -> None:
        super().setUp()
        import sparse as sp  # pylint: disable=import-error

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

    def test_is_sparse(self):
        """Test PolynomialTensor.is_sparse"""
        import sparse as sp  # pylint: disable=import-error

        with self.subTest("sparse"):
            self.assertTrue(PolynomialTensor({"+": sp.as_coo({(0,): 1})}).is_sparse())

        with self.subTest("sparse with empty key"):
            self.assertTrue(PolynomialTensor({"": 1.0, "+": sp.as_coo({(0,): 1})}).is_sparse())

        with self.subTest("dense"):
            self.assertFalse(PolynomialTensor({"+": np.array([1])}).is_sparse())

        with self.subTest("mixed"):
            self.assertFalse(
                PolynomialTensor({"+": sp.as_coo({(1,): 1}), "+-": np.eye(2)}).is_sparse()
            )

    def test_is_dense(self):
        """Test PolynomialTensor.is_dense"""
        import sparse as sp  # pylint: disable=import-error

        with self.subTest("dense"):
            self.assertTrue(PolynomialTensor({"+": np.array([1])}).is_dense())

        with self.subTest("dense with empty key"):
            self.assertTrue(PolynomialTensor({"": 1.0, "+": np.array([1])}).is_dense())

        with self.subTest("sparse"):
            self.assertFalse(PolynomialTensor({"+": sp.as_coo({(0,): 1})}).is_dense())

        with self.subTest("mixed"):
            self.assertFalse(
                PolynomialTensor({"+": sp.as_coo({(1,): 1}), "+-": np.eye(2)}).is_dense()
            )

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

    def test_to_dense(self):
        """Test PolynomialTensor.to_dense"""
        dense_tensor = PolynomialTensor(self.sparse_1).to_dense()
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

    def test_to_sparse(self):
        """Test PolynomialTensor.to_sparse"""
        import sparse as sp  # pylint: disable=import-error

        sparse_tensor = PolynomialTensor(self.og_poly).to_sparse()
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
        import sparse as sp  # pylint: disable=import-error

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
        import sparse as sp  # pylint: disable=import-error

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

    def test_compose(self):
        """Test composition of PolynomialTensor"""
        import sparse as sp  # pylint: disable=import-error

        mat1dd = self.build_matrix(4, 1)
        mat2dd = self.build_matrix(4, 2)
        mat1ddt = self.build_matrix(4, 1).transpose()
        mat2ddt = self.build_matrix(4, 2).transpose()
        mat1ds = sp.as_coo({(0,): 1, (2,): 2}, shape=(4,))
        mat2ds = sp.as_coo({(0, 0): 1, (1, 0): 2}, shape=(4, 4))
        mat1dst = mat1ds.transpose()
        mat2dst = mat2ds.transpose()

        with self.subTest("dense with dense"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1dd, "+-": mat2dd})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1ddt, "+-": mat2ddt})

            with self.subTest("compose(front=False)"):
                result = pt_a.compose(pt_b)
                expected = {
                    "": 1.0,
                    "+": mat1dd + mat1ddt,
                    "+-": mat2dd + mat2ddt,
                    "++": np.outer(mat1dd, mat1ddt),
                    "++-": np.outer(mat1ddt, mat2dd).reshape((4, 4, 4)),
                    "+-+": np.outer(mat2ddt, mat1dd).reshape((4, 4, 4)),
                    "+-+-": np.outer(mat2ddt, mat2dd).reshape((4, 4, 4, 4)),
                }
                self.assertEqual(result, PolynomialTensor(expected))

            with self.subTest("compose(front=True)"):
                result = pt_a.compose(pt_b, front=True)
                expected = {
                    "": 1.0,
                    "+": mat1dd + mat1ddt,
                    "+-": mat2dd + mat2ddt,
                    "++": np.outer(mat1ddt, mat1dd),
                    "++-": np.outer(mat1dd, mat2ddt).reshape((4, 4, 4)),
                    "+-+": np.outer(mat2dd, mat1ddt).reshape((4, 4, 4)),
                    "+-+-": np.outer(mat2dd, mat2ddt).reshape((4, 4, 4, 4)),
                }
                self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("sparse with dense"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1ds, "+-": mat2ds})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1dd, "+-": mat2dd})

            with self.subTest("compose(front=False)"):
                result = pt_a.compose(pt_b)
                expected = {
                    "": 1.0,
                    "+": mat1dd + mat1ds,
                    "+-": mat2dd + mat2ds,
                    "++": np.outer(mat1dd, mat1ds),
                    "++-": np.outer(mat1dd, mat2ds).reshape((4, 4, 4)),
                    "+-+": np.outer(mat2dd, mat1ds).reshape((4, 4, 4)),
                    "+-+-": np.outer(mat2dd, mat2ds).reshape((4, 4, 4, 4)),
                }
                self.assertEqual(result, PolynomialTensor(expected))

            with self.subTest("compose(front=True)"):
                result = pt_a.compose(pt_b, front=True)
                expected = {
                    "": 1.0,
                    "+": np.array([2, 2, 5, 4]),
                    "+-": mat2dd + mat2ds,
                    "++": np.outer(mat1ds, mat1dd),
                    "++-": np.outer(mat1ds, mat2dd).reshape((4, 4, 4)),
                    "+-+": np.outer(mat2ds, mat1dd).reshape((4, 4, 4)),
                    "+-+-": np.outer(mat2ds, mat2dd).reshape((4, 4, 4, 4)),
                }
                self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("sparse with sparse"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1ds, "+-": mat2ds})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1dst, "+-": mat2dst})

            with self.subTest("compose(front=False)"):
                result = pt_a.compose(pt_b)
                expected = {
                    "": 1.0,
                    "+": mat1ds + mat1dst,
                    "+-": mat2ds + mat2dst,
                    "++": np.outer(mat1dst, mat1ds),
                    "++-": np.outer(mat1dst, mat2ds).reshape((4, 4, 4)),
                    "+-+": np.outer(mat2dst, mat1ds).reshape((4, 4, 4)),
                    "+-+-": np.outer(mat2dst, mat2ds).reshape((4, 4, 4, 4)),
                }
                self.assertEqual(result, PolynomialTensor(expected))

            with self.subTest("compose(front=True)"):
                result = pt_a.compose(pt_b, front=True)
                expected = {
                    "": 1.0,
                    "+": mat1ds + mat1dst,
                    "+-": mat2ds + mat2dst,
                    "++": np.outer(mat1ds, mat1dst),
                    "++-": np.outer(mat1ds, mat2dst).reshape((4, 4, 4)),
                    "+-+": np.outer(mat2ds, mat1dst).reshape((4, 4, 4)),
                    "+-+-": np.outer(mat2ds, mat2dst).reshape((4, 4, 4, 4)),
                }
                self.assertEqual(result, PolynomialTensor(expected))

    def test_tensor(self):
        """Test tensoring of PolynomialTensor"""
        import sparse as sp  # pylint: disable=import-error

        mat1dd = self.build_matrix(2, 1)
        mat2dd = self.build_matrix(2, 2)
        mat1ddt = self.build_matrix(2, 1).transpose()
        mat2ddt = self.build_matrix(2, 2).transpose()
        mat1ds = sp.as_coo({(0,): 1, (1,): 2}, shape=(2,))
        mat2ds = sp.as_coo({(0, 0): 1, (1, 0): 2}, shape=(2, 2))
        mat1dst = mat1ds.transpose()
        mat2dst = mat2ds.transpose()

        zeros = np.zeros((2, 2))

        def tmp(factor1, factor2):
            return np.block(
                [
                    [[zeros, zeros], [zeros, factor1 * mat2ddt]],
                    [[zeros, zeros], [zeros, factor2 * mat2ddt]],
                    [[zeros, zeros], [zeros, zeros]],
                    [[zeros, zeros], [zeros, zeros]],
                ]
            )

        with self.subTest("dense with dense"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1dd, "+-": mat2dd})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1ddt, "+-": mat2ddt})
            result = pt_a.tensor(pt_b)
            expected = {
                "": 1.0,
                "+": np.hstack([mat1dd, mat1ddt]),
                "+-": np.block([[mat2dd, zeros], [zeros, mat2ddt]]),
                "++": sp.as_coo(
                    {(0, 2): 1, (0, 3): 2, (1, 2): 2, (1, 3): 4}, shape=(4, 4)
                ).todense(),
                "++-": tmp(1, 2),
                "+-+": sp.as_coo(
                    {
                        (0, 0, 2): 1,
                        (0, 0, 3): 2,
                        (0, 1, 2): 2,
                        (0, 1, 3): 4,
                        (1, 0, 2): 3,
                        (1, 0, 3): 6,
                        (1, 1, 2): 4,
                        (1, 1, 3): 8,
                    },
                    shape=(4, 4, 4),
                ).todense(),
                "+-+-": np.vstack([tmp(1, 2), tmp(3, 4), tmp(0, 0), tmp(0, 0)]).reshape(
                    (4, 4, 4, 4)
                ),
            }
            self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("sparse with dense"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1ds, "+-": mat2ds})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1ddt, "+-": mat2ddt})
            result = pt_a.tensor(pt_b)
            expected = {
                "": 1.0,
                "+": sp.as_coo({(0,): 1, (1,): 2, (2,): 1, (3,): 2}, shape=(4,)),
                "+-": sp.as_coo(
                    {(0, 0): 1, (1, 0): 2, (2, 2): 1, (2, 3): 3, (3, 2): 2, (3, 3): 4}, shape=(4, 4)
                ),
                "++": sp.as_coo({(0, 2): 1, (0, 3): 2, (1, 2): 2, (1, 3): 4}, shape=(4, 4)),
                "++-": sp.as_coo(tmp(1, 2)),
                "+-+": sp.as_coo(
                    {(0, 0, 2): 1, (0, 0, 3): 2, (1, 0, 2): 2, (1, 0, 3): 4}, shape=(4, 4, 4)
                ),
                "+-+-": sp.as_coo(
                    np.vstack([tmp(1, 0), tmp(2, 0), tmp(0, 0), tmp(0, 0)]).reshape((4, 4, 4, 4))
                ),
            }
            self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("sparse with sparse"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1ds, "+-": mat2ds})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1dst, "+-": mat2dst})
            result = pt_a.tensor(pt_b)
            expected = {
                "": 1.0,
                "+": sp.as_coo({(0,): 1, (1,): 2, (2,): 1, (3,): 2}, shape=(4,)),
                "+-": sp.as_coo({(0, 0): 1, (1, 0): 2, (2, 2): 1, (2, 3): 2}, shape=(4, 4)),
                "++": sp.as_coo({(0, 2): 1, (0, 3): 2, (1, 2): 2, (1, 3): 4}, shape=(4, 4)),
                "++-": sp.as_coo(
                    {(0, 2, 2): 1, (0, 2, 3): 2, (1, 2, 2): 2, (1, 2, 3): 4}, shape=(4, 4, 4)
                ),
                "+-+": sp.as_coo(
                    {(0, 0, 2): 1, (0, 0, 3): 2, (1, 0, 2): 2, (1, 0, 3): 4}, shape=(4, 4, 4)
                ),
                "+-+-": sp.as_coo(
                    {(0, 0, 2, 2): 1, (0, 0, 2, 3): 2, (1, 0, 2, 2): 2, (1, 0, 2, 3): 4},
                    shape=(4, 4, 4, 4),
                ),
            }
            self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("dense with sparse"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1ddt, "+-": mat2ddt})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1ds, "+-": mat2ds})
            result = pt_a.tensor(pt_b)
            expected = {
                "": 1.0,
                "+": sp.as_coo({(0,): 1, (1,): 2, (2,): 1, (3,): 2}, shape=(4,)),
                "+-": sp.as_coo(
                    {(0, 0): 1, (0, 1): 3, (1, 0): 2, (1, 1): 4, (2, 2): 1, (3, 2): 2}, shape=(4, 4)
                ),
                "++": sp.as_coo({(0, 2): 1, (0, 3): 2, (1, 2): 2, (1, 3): 4}, shape=(4, 4)),
                "++-": sp.as_coo(
                    {(0, 2, 2): 1, (0, 3, 2): 2, (1, 2, 2): 2, (1, 3, 2): 4}, shape=(4, 4, 4)
                ),
                "+-+": sp.as_coo(
                    {
                        (0, 0, 2): 1,
                        (0, 0, 3): 2,
                        (0, 1, 2): 3,
                        (0, 1, 3): 6,
                        (1, 0, 2): 2,
                        (1, 0, 3): 4,
                        (1, 1, 2): 4,
                        (1, 1, 3): 8,
                    },
                    shape=(4, 4, 4),
                ),
                "+-+-": sp.as_coo(
                    {
                        (0, 0, 2, 2): 1,
                        (0, 0, 3, 2): 2,
                        (0, 1, 2, 2): 3,
                        (0, 1, 3, 2): 6,
                        (1, 0, 2, 2): 2,
                        (1, 0, 3, 2): 4,
                        (1, 1, 2, 2): 4,
                        (1, 1, 3, 2): 8,
                    },
                    shape=(4, 4, 4, 4),
                ),
            }
            self.assertEqual(result, PolynomialTensor(expected))

    def test_expand(self):
        """Test expanding of PolynomialTensor"""
        import sparse as sp  # pylint: disable=import-error

        mat1dd = self.build_matrix(2, 1)
        mat2dd = self.build_matrix(2, 2)
        mat1ddt = self.build_matrix(2, 1).transpose()
        mat2ddt = self.build_matrix(2, 2).transpose()
        mat1ds = sp.as_coo({(0,): 1, (1,): 2}, shape=(2,))
        mat2ds = sp.as_coo({(0, 0): 1, (1, 0): 2}, shape=(2, 2))
        mat1dst = mat1ds.transpose()
        mat2dst = mat2ds.transpose()

        zeros = np.zeros((2, 2))

        def tmp(factor1, factor2):
            return np.block(
                [
                    [[zeros, zeros], [zeros, factor1 * mat2ddt]],
                    [[zeros, zeros], [zeros, factor2 * mat2ddt]],
                    [[zeros, zeros], [zeros, zeros]],
                    [[zeros, zeros], [zeros, zeros]],
                ]
            )

        with self.subTest("dense with dense"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1dd, "+-": mat2dd})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1ddt, "+-": mat2ddt})
            result = pt_b.expand(pt_a)
            expected = {
                "": 1.0,
                "+": np.hstack([mat1dd, mat1ddt]),
                "+-": np.block([[mat2dd, zeros], [zeros, mat2ddt]]),
                "++": sp.as_coo(
                    {(0, 2): 1, (0, 3): 2, (1, 2): 2, (1, 3): 4}, shape=(4, 4)
                ).todense(),
                "++-": tmp(1, 2),
                "+-+": sp.as_coo(
                    {
                        (0, 0, 2): 1,
                        (0, 0, 3): 2,
                        (0, 1, 2): 2,
                        (0, 1, 3): 4,
                        (1, 0, 2): 3,
                        (1, 0, 3): 6,
                        (1, 1, 2): 4,
                        (1, 1, 3): 8,
                    },
                    shape=(4, 4, 4),
                ).todense(),
                "+-+-": np.vstack([tmp(1, 2), tmp(3, 4), tmp(0, 0), tmp(0, 0)]).reshape(
                    (4, 4, 4, 4)
                ),
            }
            self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("dense with sparse"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1ds, "+-": mat2ds})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1ddt, "+-": mat2ddt})
            result = pt_b.expand(pt_a)
            expected = {
                "": 1.0,
                "+": sp.as_coo({(0,): 1, (1,): 2, (2,): 1, (3,): 2}, shape=(4,)),
                "+-": sp.as_coo(
                    {(0, 0): 1, (1, 0): 2, (2, 2): 1, (2, 3): 3, (3, 2): 2, (3, 3): 4}, shape=(4, 4)
                ),
                "++": sp.as_coo({(0, 2): 1, (0, 3): 2, (1, 2): 2, (1, 3): 4}, shape=(4, 4)),
                "++-": sp.as_coo(tmp(1, 2)),
                "+-+": sp.as_coo(
                    {(0, 0, 2): 1, (0, 0, 3): 2, (1, 0, 2): 2, (1, 0, 3): 4}, shape=(4, 4, 4)
                ),
                "+-+-": sp.as_coo(
                    np.vstack([tmp(1, 0), tmp(2, 0), tmp(0, 0), tmp(0, 0)]).reshape((4, 4, 4, 4))
                ),
            }
            self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("sparse with sparse"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1ds, "+-": mat2ds})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1dst, "+-": mat2dst})
            result = pt_b.expand(pt_a)
            expected = {
                "": 1.0,
                "+": sp.as_coo({(0,): 1, (1,): 2, (2,): 1, (3,): 2}, shape=(4,)),
                "+-": sp.as_coo({(0, 0): 1, (1, 0): 2, (2, 2): 1, (2, 3): 2}, shape=(4, 4)),
                "++": sp.as_coo({(0, 2): 1, (0, 3): 2, (1, 2): 2, (1, 3): 4}, shape=(4, 4)),
                "++-": sp.as_coo(
                    {(0, 2, 2): 1, (0, 2, 3): 2, (1, 2, 2): 2, (1, 2, 3): 4}, shape=(4, 4, 4)
                ),
                "+-+": sp.as_coo(
                    {(0, 0, 2): 1, (0, 0, 3): 2, (1, 0, 2): 2, (1, 0, 3): 4}, shape=(4, 4, 4)
                ),
                "+-+-": sp.as_coo(
                    {(0, 0, 2, 2): 1, (0, 0, 2, 3): 2, (1, 0, 2, 2): 2, (1, 0, 2, 3): 4},
                    shape=(4, 4, 4, 4),
                ),
            }
            self.assertEqual(result, PolynomialTensor(expected))

        with self.subTest("sparse with dense"):
            pt_a = PolynomialTensor({"": 1.0, "+": mat1ddt, "+-": mat2ddt})
            pt_b = PolynomialTensor({"": 1.0, "+": mat1ds, "+-": mat2ds})
            result = pt_b.expand(pt_a)
            expected = {
                "": 1.0,
                "+": sp.as_coo({(0,): 1, (1,): 2, (2,): 1, (3,): 2}, shape=(4,)),
                "+-": sp.as_coo(
                    {(0, 0): 1, (0, 1): 3, (1, 0): 2, (1, 1): 4, (2, 2): 1, (3, 2): 2}, shape=(4, 4)
                ),
                "++": sp.as_coo({(0, 2): 1, (0, 3): 2, (1, 2): 2, (1, 3): 4}, shape=(4, 4)),
                "++-": sp.as_coo(
                    {(0, 2, 2): 1, (0, 3, 2): 2, (1, 2, 2): 2, (1, 3, 2): 4}, shape=(4, 4, 4)
                ),
                "+-+": sp.as_coo(
                    {
                        (0, 0, 2): 1,
                        (0, 0, 3): 2,
                        (0, 1, 2): 3,
                        (0, 1, 3): 6,
                        (1, 0, 2): 2,
                        (1, 0, 3): 4,
                        (1, 1, 2): 4,
                        (1, 1, 3): 8,
                    },
                    shape=(4, 4, 4),
                ),
                "+-+-": sp.as_coo(
                    {
                        (0, 0, 2, 2): 1,
                        (0, 0, 3, 2): 2,
                        (0, 1, 2, 2): 3,
                        (0, 1, 3, 2): 6,
                        (1, 0, 2, 2): 2,
                        (1, 0, 3, 2): 4,
                        (1, 1, 2, 2): 4,
                        (1, 1, 3, 2): 8,
                    },
                    shape=(4, 4, 4, 4),
                ),
            }
            self.assertEqual(result, PolynomialTensor(expected))

    def test_apply(self):
        """Test PolynomialTensor.apply"""
        rand_a = np.random.random((2, 2))
        rand_b = np.random.random((2, 2))
        a = PolynomialTensor({"+-": rand_a})
        b = PolynomialTensor({"+": np.random.random(2), "+-": rand_b})

        with self.subTest("np.transpose"):
            a_transpose = PolynomialTensor.apply(np.transpose, a)
            self.assertEqual(a_transpose, PolynomialTensor({"+-": rand_a.transpose()}))

        with self.subTest("np.conjugate"):
            a_complex = 1j * a
            a_conjugate = PolynomialTensor.apply(np.conjugate, a_complex)
            self.assertEqual(a_conjugate, PolynomialTensor({"+-": -1j * rand_a}))

        with self.subTest("np.kron"):
            ab_kron = PolynomialTensor.apply(np.kron, a, b)
            self.assertEqual(ab_kron, PolynomialTensor({"+-": np.kron(rand_a, rand_b)}))

    def test_einsum(self):
        """Test PolynomialTensor.einsum"""
        import sparse as sp  # pylint: disable=import-error

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
