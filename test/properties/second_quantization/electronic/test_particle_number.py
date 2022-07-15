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

"""Test ParticleNumber Property"""

import tempfile
import warnings
from test.properties.property_test import PropertyTest

import h5py
import numpy as np

from qiskit_nature.drivers import QMolecule
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber


class TestParticleNumber(PropertyTest):
    """Test ParticleNumber Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            qmol = QMolecule()
        qmol.num_molecular_orbitals = 4
        qmol.num_alpha = 2
        qmol.num_beta = 2
        self.prop = ParticleNumber.from_legacy_driver_result(qmol)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = [self.prop.second_q_ops()["ParticleNumber"]]
        self.assertEqual(len(ops), 1)
        expected = [
            "+_0 -_0",
            "+_1 -_1",
            "+_2 -_2",
            "+_3 -_3",
            "+_4 -_4",
            "+_5 -_5",
            "+_6 -_6",
            "+_7 -_7",
        ]
        self.assertEqual([l for l, _ in ops[0].to_list()], expected)

    def test_non_singlet_occupation(self):
        """Regression test against occupation computation of non-singlet state."""
        prop = ParticleNumber(4, (2, 1), [2.0, 1.0])
        self.assertTrue(np.allclose(prop.occupation_alpha, [1.0, 1.0]))
        self.assertTrue(np.allclose(prop.occupation_beta, [1.0, 0.0]))

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
                read_prop = ParticleNumber.from_hdf5(file["ParticleNumber"])

                self.assertEqual(self.prop, read_prop)
