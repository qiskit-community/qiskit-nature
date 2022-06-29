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

"""Test Magnetization Property"""

import tempfile
import warnings
from test.properties.property_test import PropertyTest

import h5py

from qiskit_nature.drivers import QMolecule
from qiskit_nature.second_q.operator_factories.electronic import Magnetization


class TestMagnetization(PropertyTest):
    """Test Magnetization Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            qmol = QMolecule()
        qmol.num_molecular_orbitals = 4
        self.prop = Magnetization.from_legacy_driver_result(qmol)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = [self.prop.second_q_ops()["Magnetization"]]
        self.assertEqual(len(ops), 1)
        expected = [
            ("+_0 -_0", 0.5),
            ("+_1 -_1", 0.5),
            ("+_2 -_2", 0.5),
            ("+_3 -_3", 0.5),
            ("+_4 -_4", -0.5),
            ("+_5 -_5", -0.5),
            ("+_6 -_6", -0.5),
            ("+_7 -_7", -0.5),
        ]
        self.assertEqual(ops[0].to_list(), expected)

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
                read_prop = Magnetization.from_hdf5(file["Magnetization"])

                self.assertEqual(self.prop, read_prop)
