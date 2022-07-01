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

"""Test ElectronicEnergy Property"""

import json
import tempfile
from test.second_q.operator_factories.property_test import PropertyTest
from typing import cast

import h5py
import numpy as np

from qiskit_nature.second_q.drivers.second_quantization import HDF5Driver
from qiskit_nature.second_q.operator_factories.electronic import ElectronicEnergy
from qiskit_nature.second_q.operator_factories.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.second_q.operator_factories.electronic.integrals import (
    OneBodyElectronicIntegrals,
)


class TestElectronicEnergy(PropertyTest):
    """Test ElectronicEnergy Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("test_driver_hdf5.hdf5", "drivers/second_q/hdf5d")
        )
        self.prop = cast(ElectronicEnergy, driver.run().get_property(ElectronicEnergy))
        self.prop.get_electronic_integral(ElectronicBasis.MO, 1).set_truncation(2)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = [self.prop.second_q_ops()["ElectronicEnergy"]]
        self.assertEqual(len(ops), 1)
        with open(
            self.get_resource_path(
                "electronic_energy_op.json", "properties/second_q/electronic/resources"
            ),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)
        for op, expected_op in zip(ops[0].to_list(), expected):
            self.assertEqual(op[0], expected_op[0])
            self.assertTrue(np.isclose(op[1], expected_op[1]))

    def test_integral_operator(self):
        """Test integral_operator."""
        # duplicate MO integrals into AO basis for this test
        trafo = ElectronicBasisTransform(ElectronicBasis.MO, ElectronicBasis.AO, np.eye(2))
        self.prop.transform_basis(trafo)

        density = OneBodyElectronicIntegrals(ElectronicBasis.AO, (0.5 * np.eye(2), None))
        matrix_op = self.prop.integral_operator(density)

        expected = np.asarray([[-0.34436786423711596, 0.0], [0.0, 0.4515069814257469]])
        self.assertTrue(np.allclose(matrix_op._matrices[0], expected))

    def test_from_raw_integrals(self):
        """Test from_raw_integrals utility method."""
        one_body_a = np.random.random((2, 2))
        one_body_b = np.random.random((2, 2))
        two_body_aa = np.random.random((2, 2, 2, 2))
        two_body_bb = np.random.random((2, 2, 2, 2))
        two_body_ba = np.random.random((2, 2, 2, 2))

        with self.subTest("minimal SO"):
            prop = ElectronicEnergy.from_raw_integrals(ElectronicBasis.SO, one_body_a, two_body_aa)
            self.assertTrue(
                np.allclose(
                    prop.get_electronic_integral(ElectronicBasis.SO, 1)._matrices, one_body_a
                )
            )
            self.assertTrue(
                np.allclose(
                    prop.get_electronic_integral(ElectronicBasis.SO, 2)._matrices, two_body_aa
                )
            )

        with self.subTest("minimal MO"):
            prop = ElectronicEnergy.from_raw_integrals(ElectronicBasis.MO, one_body_a, two_body_aa)
            self.assertTrue(
                np.allclose(
                    prop.get_electronic_integral(ElectronicBasis.MO, 1)._matrices[0], one_body_a
                )
            )
            self.assertTrue(
                np.allclose(
                    prop.get_electronic_integral(ElectronicBasis.MO, 2)._matrices[0], two_body_aa
                )
            )

        with self.subTest("minimal MO with beta"):
            prop = ElectronicEnergy.from_raw_integrals(
                ElectronicBasis.MO,
                one_body_a,
                two_body_aa,
                h1_b=one_body_b,
                h2_bb=two_body_bb,
                h2_ba=two_body_ba,
            )
            self.assertTrue(
                np.allclose(
                    prop.get_electronic_integral(ElectronicBasis.MO, 1)._matrices[0], one_body_a
                )
            )
            self.assertTrue(
                np.allclose(
                    prop.get_electronic_integral(ElectronicBasis.MO, 1)._matrices[1], one_body_b
                )
            )
            self.assertTrue(
                np.allclose(
                    prop.get_electronic_integral(ElectronicBasis.MO, 2)._matrices[0], two_body_aa
                )
            )
            self.assertTrue(
                np.allclose(
                    prop.get_electronic_integral(ElectronicBasis.MO, 2)._matrices[1], two_body_ba
                )
            )
            self.assertTrue(
                np.allclose(
                    prop.get_electronic_integral(ElectronicBasis.MO, 2)._matrices[2], two_body_bb
                )
            )
            self.assertTrue(
                np.allclose(
                    prop.get_electronic_integral(ElectronicBasis.MO, 2)._matrices[3], two_body_ba.T
                )
            )

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
                read_prop = ElectronicEnergy.from_hdf5(file["ElectronicEnergy"])

                self.assertEqual(self.prop, read_prop)
