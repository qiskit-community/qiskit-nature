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

from __future__ import annotations

import tempfile
import unittest
from test.second_q.properties.property_test import PropertyTest

import h5py
import numpy as np
from ddt import ddt, data, unpack

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.properties import ElectronicDipoleMoment
from qiskit_nature.second_q.properties.bases import ElectronicBasis
from qiskit_nature.second_q.properties.dipole_moment import (
    DipoleMoment,
)
from qiskit_nature.second_q.properties.integrals import (
    OneBodyElectronicIntegrals,
)


@unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
@ddt
class TestElectronicDipoleMoment(PropertyTest):
    """Test ElectronicDipoleMoment Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        driver = PySCFDriver()
        self.prop = driver.run().properties.electronic_dipole_moment

    @data(
        ("DipoleMomentX", {}),
        ("DipoleMomentY", {}),
        (
            "DipoleMomentZ",
            {
                "+_0 -_0": 0.6944743538354734,
                "+_0 -_1": 0.9278334722175678,
                "+_1 -_0": 0.9278334722175678,
                "+_1 -_1": 0.6944743538354735,
                "+_2 -_2": 0.6944743538354734,
                "+_2 -_3": 0.9278334722175678,
                "+_3 -_2": 0.9278334722175678,
                "+_3 -_3": 0.6944743538354735,
            },
        ),
    )
    @unpack
    def test_second_q_ops(self, key: str, expected_op_data: dict[str, float]):
        """Test second_q_ops."""
        op = self.prop.second_q_ops()[key]
        self.assertEqual(len(op), len(expected_op_data))
        for (key1, val1), (key2, val2) in zip(op.items(), expected_op_data.items()):
            self.assertEqual(key1, key2)
            self.assertTrue(np.isclose(np.abs(val1), val2))

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


if __name__ == "__main__":
    unittest.main()
