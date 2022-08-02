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

import unittest
from typing import Dict
from test import QiskitNatureTestCase
from ddt import ddt, data
import numpy as np
from qiskit_nature.second_q.operators import PolynomialTensor


def gen_expected_prod_poly(og_poly, dim_size, other):
    """Generate expected product Polynomial Tensors"""

    expected_poly = {}
    for key, value in og_poly.items():
        num_dim = len(key)
        expected_poly[key] = (np.arange(1, dim_size ** num_dim + 1) * other).reshape((dim_size,) * num_dim)

    return expected_poly

def gen_expected_transpose_poly(og_poly, dim_size):
    """Generate expected transpose of Polynomial Tensor"""

    expected_poly = {}
    for key, value in og_poly.items():
        num_dim = len(key)
        expected_poly[key] = np.arange(1, dim_size ** num_dim + 1).reshape((dim_size,) * num_dim).transpose()

    return expected_poly

def gen_expected_conjugate_poly(og_poly, dim_size):
    """Generate expected transpose of Polynomial Tensor"""

    expected_poly = {}
    for key, value in og_poly.items():
        num_dim = len(key)
        expected_poly[key] = np.arange(1, dim_size ** num_dim + 1).reshape((dim_size,) * num_dim).conjugate()

    return expected_poly

@ddt
class TestPolynomialTensor(QiskitNatureTestCase):
    """Tests for PolynomialTensor class"""

    def setUp(self) -> None:
        super().setUp()

        self.og_poly: Dict[str, np.ndarray] = {
            "+-": np.arange(1, 17).reshape(4, 4),
            "+++-": np.arange(1, 257).reshape(4, 4, 4, 4),
        }

        for value in self.og_poly.values():
            self.dim_size = np.shape(value)

        self.expected_sum_poly: Dict[str, np.ndarray] = {
            "+-": (np.arange(1, 17) * 2).reshape(4, 4),
            "+++-": (np.arange(1, 257) * 2).reshape(4, 4, 4, 4),
        }


    def test_init_dict(self):
        """Test for input type in Polynomial Tensor class"""

        if self.assertRaises(TypeError):
            PolynomialTensor(self.og_poly)

    def test_init_dimensions(self):
        """Test for input matrix dimensions in Polynomial Tensor"""

        if self.assertRaisesRegex(ValueError, "Dimensions of value matrices in data dictionary are not identical."):
            PolynomialTensor(self.og_poly)

    @data(2, 3, 4)
    def test_mul(self, other):
        """Test for scalar multiplication"""

        result = PolynomialTensor(self.og_poly).mul(other)
        expected_prod_poly = gen_expected_prod_poly(self.og_poly, self.dim_size[0], other)
        self.assertEqual(result, PolynomialTensor(expected_prod_poly))

    def test_add(self):
        """Test for addition of Polynomial Tensors"""

        result = PolynomialTensor(self.og_poly).add(PolynomialTensor(self.og_poly))
        self.assertEqual(result, PolynomialTensor(self.expected_sum_poly))

    def test_conjugate(self):
        """Test for conjugate of Polynomial Tensor"""

        result = PolynomialTensor(self.og_poly).conjugate()
        expected_conjugate_poly = gen_expected_conjugate_poly(self.og_poly, self.dim_size[0])
        self.assertEqual(result, PolynomialTensor(expected_conjugate_poly))

    def test_transpose(self):
        """Test for transpose of Polynomial Tensor"""

        result = PolynomialTensor(self.og_poly).transpose()
        expected_transpose_poly = gen_expected_transpose_poly(self.og_poly, self.dim_size[0])
        self.assertEqual(result, PolynomialTensor(expected_transpose_poly))


if __name__ == "__main__":
    unittest.main()
