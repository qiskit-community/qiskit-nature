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
from ddt import ddt, idata
import numpy as np
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

        self.sample_poly_1 = {
            "": 1.0,
            "++": self.build_matrix(4, 1),
            "+-": self.build_matrix(4, 2),
            "++--": self.build_matrix(4, 4),
        }

        self.sample_poly_2 = {
            "": 1.0,
            "+": self.build_matrix(4, 1),
            "+-": self.build_matrix(4, 2),
            "++--": np.arange(1, 13).reshape(1, 2, 3, 2),
        }

        self.sample_poly_3 = {
            "": 1.0,
            "+": self.build_matrix(4, 1),
            "+-": self.build_matrix(2, 2),
            "++--": self.build_matrix(4, 4),
        }

        self.sample_poly_4 = {
            "": 1.0,
            "+": self.build_matrix(2, 1),
            "+-": self.build_matrix(2, 2),
            "++--": self.build_matrix(2, 4),
        }

        self.sample_poly_5 = {
            "": 1.0,
            "+": self.build_matrix(2, 1),
            "+-": self.build_matrix(2, 2),
        }

        self.expected_conjugate_poly = {
            "": 1.0,
            "+": self.build_matrix(4, 1).conjugate(),
            "+-": self.build_matrix(4, 2).conjugate(),
            "++--": self.build_matrix(4, 4).conjugate(),
        }

        self.expected_transpose_poly = {
            "": 1.0,
            "+": self.build_matrix(4, 1).transpose(),
            "+-": self.build_matrix(4, 2).transpose(),
            "++--": self.build_matrix(4, 4).transpose(),
        }

        self.expected_sum_poly = {
            "": 2.0,
            "+": np.add(self.build_matrix(4, 1), self.build_matrix(4, 1)),
            "+-": np.add(self.build_matrix(4, 2), self.build_matrix(4, 2)),
            "++--": np.add(self.build_matrix(4, 4), self.build_matrix(4, 4)),
        }

        self.expected_compose_poly = {
            "": 1.0,
            "+": np.multiply(self.build_matrix(4, 1).transpose(), self.build_matrix(4, 1)),
            "+-": np.matmul(self.build_matrix(4, 2).transpose(), self.build_matrix(4, 2)),
            "++--": np.matmul(self.build_matrix(4, 4).transpose(), self.build_matrix(4, 4)),
        }

        self.expected_compose_front_poly = {
            "": 1.0,
            "+": np.multiply(self.build_matrix(4, 1), self.build_matrix(4, 1).transpose()),
            "+-": np.matmul(self.build_matrix(4, 2), self.build_matrix(4, 2).transpose()),
            "++--": np.matmul(self.build_matrix(4, 4), self.build_matrix(4, 4).transpose()),
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

    @staticmethod
    def build_matrix(dim_size, num_dim, val=1):
        """Build dictionary value matrix"""
        return (np.arange(1, dim_size**num_dim + 1) * val).reshape((dim_size,) * num_dim)

    def test_init(self):
        """Test for errors in constructor for PolynomialTensor"""
        with self.assertRaisesRegex(
            ValueError,
            r"Data key .* of length \d does not match data value matrix of dimensions \(\d+, *\)",
        ):
            _ = PolynomialTensor(self.sample_poly_1)

        with self.assertRaisesRegex(
            ValueError, r"For key (.*): dimensions of value matrix are not identical \(\d+, .*\)"
        ):
            _ = PolynomialTensor(self.sample_poly_2)

    def test_get_item(self):
        """Test for getting value matrices corresponding to keys in PolynomialTensor"""
        og_poly_tensor = PolynomialTensor(self.og_poly)
        for key, value in self.og_poly.items():
            np.testing.assert_array_equal(value, og_poly_tensor[key])

    def test_len(self):
        """Test for the length of PolynomialTensor"""
        length = len(PolynomialTensor(self.sample_poly_4))
        exp_len = 4
        self.assertEqual(exp_len, length)

    def test_iter(self):
        """Test for the iterator of PolynomialTensor"""
        og_poly_tensor = PolynomialTensor(self.og_poly)
        exp_iter = [key for key, _ in self.og_poly.items()]
        self.assertEqual(exp_iter, list(iter(og_poly_tensor)))

    @idata(np.linspace(0, 3, 5))
    def test_mul(self, other):
        """Test for scalar multiplication"""
        expected_prod_poly = {
            "": 1.0 * other,
            "+": self.build_matrix(4, 1, other),
            "+-": self.build_matrix(4, 2, other),
            "++--": self.build_matrix(4, 4, other),
        }

        result = PolynomialTensor(self.og_poly) * other
        self.assertEqual(result, PolynomialTensor(expected_prod_poly))

        with self.assertRaisesRegex(TypeError, r"other .* must be a number"):
            _ = PolynomialTensor(self.og_poly) * PolynomialTensor(self.og_poly)

    def test_add(self):
        """Test for addition of PolynomialTensor"""
        result = PolynomialTensor(self.og_poly) + PolynomialTensor(self.og_poly)
        self.assertEqual(result, PolynomialTensor(self.expected_sum_poly))

        with self.assertRaisesRegex(
            TypeError, "Incorrect argument type: other should be PolynomialTensor"
        ):
            _ = PolynomialTensor(self.og_poly) + 5

    def test_conjugate(self):
        """Test for conjugate of PolynomialTensor"""
        result = PolynomialTensor(self.og_poly).conjugate()
        self.assertEqual(result, PolynomialTensor(self.expected_conjugate_poly))

    def test_transpose(self):
        """Test for transpose of PolynomialTensor"""
        result = PolynomialTensor(self.og_poly).transpose()
        self.assertEqual(result, PolynomialTensor(self.expected_transpose_poly))

    def test_compose(self):
        """Test composition of PolynomialTensor"""
        pt_a = PolynomialTensor(self.og_poly)
        pt_b = PolynomialTensor(self.expected_transpose_poly)

        with self.subTest("compose(front=False)"):
            result = pt_a.compose(pt_b)
            self.assertEqual(result, PolynomialTensor(self.expected_compose_poly))

        with self.subTest("compose(front=True)"):
            result = pt_a.compose(pt_b, front=True)
            self.assertEqual(result, PolynomialTensor(self.expected_compose_front_poly))

    def test_tensor(self):
        """Test tensoring of PolynomialTensor"""
        p_t = PolynomialTensor(self.og_poly)
        result = PolynomialTensor(self.kronecker).tensor(p_t)
        self.assertEqual(result, PolynomialTensor(self.expected_tensor_poly))

    def test_expand(self):
        """Test expanding of PolynomialTensor"""
        p_t = PolynomialTensor(self.og_poly)
        result = p_t.expand(PolynomialTensor(self.kronecker))
        self.assertEqual(result, PolynomialTensor(self.expected_tensor_poly))

    def test_einsum(self):
        """Test PolynomialTensor.einsum"""
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


if __name__ == "__main__":
    unittest.main()
