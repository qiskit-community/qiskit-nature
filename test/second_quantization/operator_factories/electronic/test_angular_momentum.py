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
from test.properties.property_test import PropertyTest

import h5py

from qiskit_nature.drivers import QMolecule
from qiskit_nature.second_quantization.operator_factories.electronic import AngularMomentum
from qiskit_nature.second_quantization.operators import FermionicOp


class TestAngularMomentum(PropertyTest):
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
        op = self.prop.second_q_ops()["AngularMomentum"]
        with open(
            self.get_resource_path(
                "angular_momentum_op.json", "properties/second_quantization/electronic/resources"
            ),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)
            expected_op = FermionicOp(expected).simplify()
        self.assertSetEqual(frozenset(op.to_list("dense")), frozenset(expected_op.to_list("dense")))

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
                read_prop = AngularMomentum.from_hdf5(file["AngularMomentum"])

                self.assertEqual(self.prop, read_prop)
