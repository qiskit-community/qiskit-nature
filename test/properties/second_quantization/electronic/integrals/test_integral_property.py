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

"""Test IntegralProperty"""

from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    IntegralProperty,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)


class TestIntegralProperty(QiskitNatureTestCase):
    """Test IntegralProperty Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        self.ints_1_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (np.eye(2), None))
        self.ints_1_mo = OneBodyElectronicIntegrals(ElectronicBasis.MO, (np.eye(2), None))
        self.ints_2_ao = TwoBodyElectronicIntegrals(
            ElectronicBasis.AO, (np.ones((2, 2, 2, 2)), None, None, None)
        )
        self.ints_2_mo = TwoBodyElectronicIntegrals(
            ElectronicBasis.MO, (np.ones((2, 2, 2, 2)), None, None, None)
        )
        self.prop = IntegralProperty(
            "test", [self.ints_1_ao, self.ints_1_mo, self.ints_2_ao, self.ints_2_mo]
        )

    def test_init(self):
        """Test construction."""
        self.assertEqual(
            self.prop._electronic_integrals,
            {
                ElectronicBasis.AO: {1: self.ints_1_ao, 2: self.ints_2_ao},
                ElectronicBasis.MO: {1: self.ints_1_mo, 2: self.ints_2_mo},
            },
        )

    def test_get_electronic_int(self):
        """Test get_electronic_integral."""
        with self.subTest("Exists"):
            ints = self.prop.get_electronic_integral(ElectronicBasis.AO, 1)
            self.assertEqual(ints, self.ints_1_ao)
        with self.subTest("None"):
            ints = self.prop.get_electronic_integral(ElectronicBasis.AO, 3)
            self.assertIsNone(ints)
            ints = self.prop.get_electronic_integral(ElectronicBasis.SO, 1)
            self.assertIsNone(ints)

    def test_transform_basis(self):
        """Test transform_basis."""
        trafo = ElectronicBasisTransform(ElectronicBasis.AO, ElectronicBasis.MO, 2.0 * np.eye(2))

        self.prop.transform_basis(trafo)

        for basis, factor in zip((ElectronicBasis.AO, ElectronicBasis.MO), (1, 4)):
            for matrix in self.prop.get_electronic_integral(basis, 1)._matrices:
                self.assertTrue(np.allclose(matrix, factor * np.eye(2)))

        for basis, factor in zip((ElectronicBasis.AO, ElectronicBasis.MO), (1, 16)):
            for matrix in self.prop.get_electronic_integral(basis, 2)._matrices:
                self.assertTrue(np.allclose(matrix, factor * np.ones((2, 2, 2, 2))))

    def test_second_q_ops(self):
        """Test second_q_ops."""
        second_q_ops = self.prop.second_q_ops()
        expected = [
            ("+_0 -_1 +_2 -_3", (1 + 0j)),
            ("+_0 -_1 -_2 +_3", (-1 + 0j)),
            ("+_0 -_1 +_3 -_3", (1 + 0j)),
            ("+_0 -_1 +_2 -_2", (1 + 0j)),
            ("-_0 +_1 +_2 -_3", (-1 + 0j)),
            ("-_0 +_1 -_2 +_3", (1 + 0j)),
            ("-_0 +_1 +_3 -_3", (-1 + 0j)),
            ("-_0 +_1 +_2 -_2", (-1 + 0j)),
            ("+_3 -_3", (1 + 0j)),
            ("+_2 -_2", (1 + 0j)),
            ("+_1 -_1 +_2 -_3", (1 + 0j)),
            ("+_1 -_1 -_2 +_3", (-1 + 0j)),
            ("+_1 -_1", (1 + 0j)),
            ("+_1 -_1 +_3 -_3", (1 + 0j)),
            ("+_1 -_1 +_2 -_2", (1 + 0j)),
            ("+_0 -_0 +_2 -_3", (1 + 0j)),
            ("+_0 -_0 -_2 +_3", (-1 + 0j)),
            ("+_0 -_0", (1 + 0j)),
            ("+_0 -_0 +_3 -_3", (1 + 0j)),
            ("+_0 -_0 +_2 -_2", (1 + 0j)),
        ]
        self.assertEqual(second_q_ops[0].to_list(), expected)
