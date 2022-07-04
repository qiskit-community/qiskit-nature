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

"""Test IntegralProperty"""

import json
import tempfile
from test.second_q.operator_factories.property_test import PropertyTest

import h5py
import numpy as np

from qiskit_nature.second_q.operator_factories.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.second_q.operator_factories.electronic.integrals import (
    IntegralProperty,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)


class TestIntegralProperty(PropertyTest):
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
            matrices = self.prop.get_electronic_integral(basis, 1)._matrices
            self.assertTrue(np.allclose(matrices[0], factor * np.eye(2)))
            for mat in matrices[1:]:
                self.assertIsNone(mat)

        for basis, factor in zip((ElectronicBasis.AO, ElectronicBasis.MO), (1, 16)):
            matrices = self.prop.get_electronic_integral(basis, 2)._matrices
            self.assertTrue(np.allclose(matrices[0], factor * np.ones((2, 2, 2, 2))))
            for mat in matrices[1:]:
                self.assertIsNone(mat)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        second_q_ops = [self.prop.second_q_ops()["test"]]
        with open(
            self.get_resource_path(
                "integral_property_op.json",
                "second_q/operator_factories/electronic/integrals/resources",
            ),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)
        for op, expected_op in zip(second_q_ops[0].to_list(), expected):
            self.assertEqual(op[0], expected_op[0])
            self.assertTrue(np.isclose(op[1], expected_op[1]))

    def test_to_hdf5(self):
        """Test to_hdf5."""
        with tempfile.TemporaryFile() as tmp_file:
            with h5py.File(tmp_file, "w") as file:
                self.prop.to_hdf5(file)

    def test_from_hdf5(self):
        """Test from_hdf5."""
        with tempfile.TemporaryFile() as tmp_file:
            with h5py.File(tmp_file, "w") as file:
                self.prop.to_hdf5(file)

            with h5py.File(tmp_file, "r") as file:
                read_prop = IntegralProperty.from_hdf5(file["test"])

                self.assertDictEqual(self.prop._shift, read_prop._shift)

                for f_int, s_int in zip(iter(self.prop), iter(read_prop)):
                    self.assertEqual(f_int, s_int)
