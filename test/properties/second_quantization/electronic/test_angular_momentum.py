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

"""Test AngularMomentum Property"""

import json
import tempfile
import warnings
from test import QiskitNatureTestCase

import h5py
import numpy as np

from qiskit_nature.drivers import QMolecule
from qiskit_nature.properties.second_quantization.electronic import AngularMomentum


class TestAngularMomentum(QiskitNatureTestCase):
    """Test AngularMomentum Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            qmol = QMolecule()
        qmol.num_molecular_orbitals = 4
        self.prop = AngularMomentum.from_legacy_driver_result(qmol)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = self.prop.second_q_ops()
        self.assertEqual(len(ops), 1)
        with open(
            self.get_resource_path(
                "angular_momentum_op.json", "properties/second_quantization/electronic/resources"
            ),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)
        for op, expected_op in zip(ops[0].to_list(), expected):
            self.assertEqual(op[0], expected_op[0])
            self.assertTrue(np.isclose(op[1], expected_op[1]))

    def test_to_hdf5(self):
        """Test to_hdf5."""
        with tempfile.TemporaryFile() as tmp_file:
            with h5py.File(tmp_file, "w") as file:
                self.prop.to_hdf5(file)

            with h5py.File(tmp_file, "r") as file:
                count = 0

                for name, group in file.items():
                    count += 1
                    self.assertEqual(name, "AngularMomentum")
                    self.assertEqual(group.attrs["num_spin_orbitals"], 8)
                    self.assertIsNone(group.attrs.get("spin", None))

                self.assertEqual(count, 1)

    def test_from_hdf5(self):
        """Test from_hdf5."""
        self.skipTest("Testing via ElectronicStructureResult tests.")
