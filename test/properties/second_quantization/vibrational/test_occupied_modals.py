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

"""Test OccupiedModals Property"""

import tempfile
import warnings
from test.properties.property_test import PropertyTest

import h5py

from qiskit_nature.drivers import WatsonHamiltonian
from qiskit_nature.properties.second_quantization.vibrational import OccupiedModals
from qiskit_nature.properties.second_quantization.vibrational.bases import HarmonicBasis


class TestOccupiedModals(PropertyTest):
    """Test OccupiedModals Property"""

    def setUp(self):
        """Setup basis."""
        super().setUp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            watson = WatsonHamiltonian([], -1)
        basis = HarmonicBasis([2, 3, 4])
        self.prop = OccupiedModals.from_legacy_driver_result(watson)
        self.prop.basis = basis

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = [self.prop.second_q_ops()["0"]]
        expected = [
            [("NIIIIIIII", (1 + 0j)), ("INIIIIIII", (1 + 0j))],
            [("IINIIIIII", (1 + 0j)), ("IIINIIIII", (1 + 0j)), ("IIIINIIII", (1 + 0j))],
            [
                ("IIIIINIII", (1 + 0j)),
                ("IIIIIINII", (1 + 0j)),
                ("IIIIIIINI", (1 + 0j)),
                ("IIIIIIIIN", (1 + 0j)),
            ],
        ]
        for op, expected_op_list in zip(ops, expected):
            self.assertEqual(op.to_list(), expected_op_list)

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
                read_prop = OccupiedModals.from_hdf5(file["OccupiedModals"])

                self.assertEqual(self.prop, read_prop)
