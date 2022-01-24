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
from test import QiskitNatureTestCase

import h5py
import numpy as np

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.properties.second_quantization.electronic import ElectronicDipoleMoment
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.dipole_moment import DipoleMoment
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
)


class TestElectronicDipoleMoment(QiskitNatureTestCase):
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
        ops = self.prop.second_q_ops()
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

            with h5py.File(tmp_file, "r") as file:
                count = 0

                for name, group in file.items():
                    count += 1
                    self.assertEqual(name, "ElectronicDipoleMoment")
                    self.assertEqual(group.attrs["reverse_dipole_sign"], False)
                    self.assertTrue(
                        np.allclose(group.attrs["nuclear_dipole_moment"], [0.0, 0.0, 1.38894871])
                    )

                    axis_x = DipoleMoment.from_hdf5(group["DipoleMomentX"])
                    for ints in iter(axis_x):
                        for mat in ints._matrices:
                            count += 1
                            self.assertTrue(np.allclose(mat, np.zeros((2, 2))))

                    axis_y = DipoleMoment.from_hdf5(group["DipoleMomentY"])
                    for ints in iter(axis_y):
                        for mat in ints._matrices:
                            count += 1
                            self.assertTrue(np.allclose(mat, np.zeros((2, 2))))

                    expected_z = np.asarray([[-0.69447435, 0.92783347], [0.92783347, -0.69447435]])
                    axis_z = DipoleMoment.from_hdf5(group["DipoleMomentZ"])
                    for ints in iter(axis_z):
                        for mat in ints._matrices:
                            count += 1
                            self.assertTrue(np.allclose(mat, expected_z))

                    self.assertTrue("dipole_shift" in group.keys())

                self.assertEqual(count, 7)

    def test_from_hdf5(self):
        """Test from_hdf5."""
        self.skipTest("Testing via ElectronicStructureResult tests.")


class TestDipoleMoment(QiskitNatureTestCase):
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

            with h5py.File(tmp_file, "r") as file:
                count = 0

                for name, group in file.items():
                    count += 1
                    self.assertEqual(name, "DipoleMomentX")
                    self.assertEqual(group.attrs["axis"], "x")

                    for ints in group["electronic_integrals"]["AO"][
                        "OneBodyElectronicIntegrals"
                    ].values():
                        count += 1
                        self.assertTrue(np.allclose(ints[...], random))

                    self.assertTrue("shift" in group.keys())

                self.assertEqual(count, 3)

    def test_from_hdf5(self):
        """Test from_hdf5."""
        self.skipTest("Testing via ElectronicStructureResult tests.")
