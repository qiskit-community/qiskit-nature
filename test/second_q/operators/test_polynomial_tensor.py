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

"""Test for Polynomial Tensor"""

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

    @staticmethod
    def build_matrix(dim_size, num_dim, val=1):
        """Build dictionary value matrix"""

        return (np.arange(1, dim_size**num_dim + 1) * val).reshape((dim_size,) * num_dim)

    def test_init(self):
        """Test for errors in constructor for Polynomial Tensor"""

        with self.assertRaisesRegex(
            ValueError,
            r"Data key .* of length \d does not match data value matrix of dimensions \(\d+, *\)",
        ):
            _ = PolynomialTensor(self.sample_poly_1, register_length=4)

        with self.assertRaisesRegex(
            ValueError,
            r"For key (.*): dimensions of value matrix are not identical \(\d+, .*\)"
        ):
            _ = PolynomialTensor(self.sample_poly_2, register_length=4)

        with self.assertRaisesRegex(
            ValueError,
            r"Dimensions of value matrices in data dictionary do not match the provided register length, \d"
        ):
            _ = PolynomialTensor(self.sample_poly_3, register_length=4)

    def test_get_item(self):
        pass

    def test_len(self):
        pass

    def test_iter(self):
        pass

    def test_register_length(self):
        pass

    @idata(np.linspace(0, 3, 5))
    def test_mul(self, other):
        """Test for scalar multiplication"""

        expected_prod_poly = {
            "": 1.0 * other,
            "+": self.build_matrix(4, 1, other),
            "+-": self.build_matrix(4, 2, other),
            "++--": self.build_matrix(4, 4, other),
        }

        result = PolynomialTensor(self.og_poly, 4) * other
        self.assertEqual(result, PolynomialTensor(expected_prod_poly, 4))

        with self.assertRaisesRegex(TypeError, r"other .* must be a number"):
            _ = PolynomialTensor(self.og_poly, 4) * PolynomialTensor(self.og_poly, 4)

    def test_add(self):
        """Test for addition of Polynomial Tensors"""

        result = PolynomialTensor(self.og_poly, 4) + PolynomialTensor(self.og_poly, 4)
        self.assertEqual(result, PolynomialTensor(self.expected_sum_poly, 4))

        with self.assertRaisesRegex(
            TypeError, "Incorrect argument type: other should be PolynomialTensor"
        ):
            _ = PolynomialTensor(self.og_poly, 4) + 5

        with self.assertRaisesRegex(
            ValueError,
            r"For key (.*): corresponding data value of shape \(\d+, *\) "
            r"does not match other value matrix of shape \(\d+, *\)",
        ):
            _ = PolynomialTensor(self.og_poly, 4) + PolynomialTensor(self.sample_poly_4, 2)

    def test_conjugate(self):
        """Test for conjugate of Polynomial Tensor"""

        result = PolynomialTensor(self.og_poly, 4).conjugate()
        self.assertEqual(result, PolynomialTensor(self.expected_conjugate_poly, 4))

    def test_transpose(self):
        """Test for transpose of Polynomial Tensor"""

        result = PolynomialTensor(self.og_poly, 4).transpose()
        self.assertEqual(result, PolynomialTensor(self.expected_transpose_poly, 4))


if __name__ == "__main__":
    unittest.main()
