# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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
import numpy as np

from test import QiskitNatureTestCase

from qiskit_nature.drivers.second_quantization import HDF5Driver, QMolecule
from qiskit_nature.properties.electronic_structure import ElectronicEnergy


class TestElectronicEnergy(QiskitNatureTestCase):
    def setUp(self):
        """Setup."""
        super().setUp()
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "test_driver_hdf5.hdf5", "drivers/second_quantization/hdf5d"
            )
        )
        qmol = driver.run()
        self.prop = ElectronicEnergy.from_driver_result(qmol)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = self.prop.second_q_ops()
        assert len(ops) == 1
        with open(
            self.get_resource_path(
                "electronic_energy_op.json", "properties/electronic_structure/resources"
            ),
            "r",
        ) as f:
            expected = json.load(f)
        for op, expected_op in zip(ops[0].to_list(), expected):
            assert op[0] == expected_op[0]
            assert np.isclose(op[1], expected_op[1])
