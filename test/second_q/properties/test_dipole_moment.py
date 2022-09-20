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

"""Test ElectronicDipoleMoment Property"""

from __future__ import annotations

import unittest
from test.second_q.properties.property_test import PropertyTest

import numpy as np
from ddt import ddt, data, unpack

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver


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
        ("XDipole", {}),
        ("YDipole", {}),
        (
            "ZDipole",
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


if __name__ == "__main__":
    unittest.main()
