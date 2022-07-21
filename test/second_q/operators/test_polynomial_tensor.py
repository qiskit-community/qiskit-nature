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
from functools import lru_cache
from test import QiskitNatureTestCase
import numpy as np
from qiskit_nature.second_q.operators import PolynomialTensor
from qiskit_nature.drivers import UnitsType
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.properties.bases.electronic_basis import ElectronicBasis


@lru_cache
def driver_results():
    """Caching driver results"""
    _driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 0.735", unit=UnitsType.ANGSTROM, basis="sto3g")
    _elec_energy = _driver.run().get_property("ElectronicEnergy")

    _one = _elec_energy.get_electronic_integral(ElectronicBasis.MO, 1)
    one_matrix = _one.to_spin()

    _two = _elec_energy.get_electronic_integral(ElectronicBasis.MO, 2)
    two_matrix = np.einsum("ijkl->iklj", _two.to_spin())

    og_poly = {"+-": one_matrix, "++--": two_matrix}

    return og_poly


class TestPolynomialTensor(QiskitNatureTestCase):
    """Tests for PolynomialTensor class"""

    def setUp(self) -> None:
        super().setUp()

        self.og_poly = driver_results()

        self.expected_poly = {
            "+-": np.array(
                [
                    [-2.51267815, 0.0, 0.0, 0.0],
                    [0.0, -0.94379201, 0.0, 0.0],
                    [0.0, 0.0, -2.51267815, 0.0],
                    [0.0, 0.0, 0.0, -0.94379201],
                ]
            ),
            "++--": np.array(
                [
                    [
                        [
                            [-0.67571015, 0.0, 0.0, 0.0],
                            [0.0, -0.1809312, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, -0.66458173, 0.0, 0.0],
                            [-0.1809312, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, -0.67571015, 0.0],
                            [0.0, 0.0, 0.0, -0.1809312],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, -0.66458173],
                            [0.0, 0.0, -0.1809312, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                    ],
                    [
                        [
                            [0.0, -0.1809312, 0.0, 0.0],
                            [-0.66458173, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [-0.1809312, 0.0, 0.0, 0.0],
                            [0.0, -0.69857372, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, -0.1809312],
                            [0.0, 0.0, -0.66458173, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, -0.1809312, 0.0],
                            [0.0, 0.0, 0.0, -0.69857372],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [-0.67571015, 0.0, 0.0, 0.0],
                            [0.0, -0.1809312, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, -0.66458173, 0.0, 0.0],
                            [-0.1809312, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, -0.67571015, 0.0],
                            [0.0, 0.0, 0.0, -0.1809312],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, -0.66458173],
                            [0.0, 0.0, -0.1809312, 0.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, -0.1809312, 0.0, 0.0],
                            [-0.66458173, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [-0.1809312, 0.0, 0.0, 0.0],
                            [0.0, -0.69857372, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, -0.1809312],
                            [0.0, 0.0, -0.66458173, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, -0.1809312, 0.0],
                            [0.0, 0.0, 0.0, -0.69857372],
                        ],
                    ],
                ]
            ),
        }

    def test_mul(self):
        """Test for scalar multiplication"""

        result = PolynomialTensor(self.og_poly).mul(2)
        self.assertEqual(result, PolynomialTensor(self.expected_poly))

    def test_add(self):
        """Test for addition of Polynomial Tensors"""

        result = PolynomialTensor(self.og_poly).add(PolynomialTensor(self.og_poly))
        self.assertEqual(result, PolynomialTensor(self.expected_poly))

    def test_conjugate(self):
        """Test for conjugate of Polynomial Tensor"""
        pass

    def transpose(self):
        """Test for transpose of Polynomial Tensor"""
        pass


if __name__ == "__main__":
    unittest.main()
