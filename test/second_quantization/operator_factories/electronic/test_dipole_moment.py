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

"""Test DipoleMoment Property"""

import json
import tempfile
from test.properties.property_test import PropertyTest

import h5py
import numpy as np

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.second_quantization.operator_factories.electronic import ElectronicDipoleMoment
from qiskit_nature.second_quantization.operator_factories.electronic.bases import ElectronicBasis
from qiskit_nature.second_quantization.operator_factories.electronic.dipole_moment import (
    DipoleMoment,
)
from qiskit_nature.second_quantization.operator_factories.electronic.integrals import (
    OneBodyElectronicIntegrals,
)


class TestElectronicDipoleMoment(PropertyTest):
    """Test ElectronicDipoleMoment Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "test_driver_hdf5.hdf5", "drivers/second_quantization/hdf5d"
            )
        )
        self.prop = driver.run().get_property(ElectronicDipoleMoment)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = [
            self.prop.second_q_ops()["DipoleMomentX"],
            self.prop.second_q_ops()["DipoleMomentY"],
            self.prop.second_q_ops()["DipoleMomentZ"],
        ]
        self.assertEqual(len(ops), 3)
        with open(
            self.get_resource_path(
                "dipole_moment_ops.json", "properties/second_quantization/electronic/resources"
            ),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)
        for op, expected_op in zip(ops, expected):
            for truth, exp in zip(op.to_list(), expected_op):
                self.assertEqual(truth[0], exp[0])
                self.assertTrue(np.isclose(truth[1], exp[1]))

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
                read_prop = ElectronicDipoleMoment.from_hdf5(file["ElectronicDipoleMoment"])

                self.assertEqual(self.prop, read_prop)


class TestDipoleMoment(PropertyTest):
    """Test DipoleMoment Property"""

    def test_integral_operator(self):
        """Test integral_operator."""
        random = np.random.random((4, 4))
        prop = DipoleMoment("x", [OneBodyElectronicIntegrals(ElectronicBasis.AO, (random, None))])
        matrix_op = prop.integral_operator(None)
        # the matrix-operator of the dipole moment is unaffected by the density!
        self.assertTrue(np.allclose(random, matrix_op._matrices[0]))

    def test_to_hdf5(self):
        """Test to_hdf5."""
        random = np.random.random((4, 4))
        prop = DipoleMoment("x", [OneBodyElectronicIntegrals(ElectronicBasis.AO, (random, None))])

        with tempfile.TemporaryFile() as tmp_file:
            with h5py.File(tmp_file, "w") as file:
                prop.to_hdf5(file)

    def test_from_hdf5(self):
        """Test from_hdf5."""
        random = np.random.random((4, 4))
        prop = DipoleMoment("x", [OneBodyElectronicIntegrals(ElectronicBasis.AO, (random, None))])

        with tempfile.TemporaryFile() as tmp_file:
            with h5py.File(tmp_file, "w") as file:
                prop.to_hdf5(file)

            with h5py.File(tmp_file, "r") as file:
                read_prop = DipoleMoment.from_hdf5(file["DipoleMomentX"])

                self.assertEqual(prop, read_prop)
