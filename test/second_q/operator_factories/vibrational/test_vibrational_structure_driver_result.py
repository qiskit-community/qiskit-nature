# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test VibrationalStructureDriverResult Property"""

from test.second_q.operator_factories.property_test import PropertyTest

import h5py

from qiskit_nature.second_q.drivers import GaussianForcesDriver
from qiskit_nature.second_q.operator_factories.vibrational import (
    VibrationalStructureDriverResult,
)
from qiskit_nature.second_q.operator_factories.vibrational.bases import HarmonicBasis


class TestVibrationalStructureDriverResult(PropertyTest):
    """Test VibrationalStructureDriverResult Property"""

    def setUp(self) -> None:
        """Setup expected object."""
        super().setUp()

        driver = GaussianForcesDriver(
            logfile=self.get_resource_path(
                "test_driver_gaussian_log_C01.txt", "drivers/second_q/gaussiand"
            )
        )
        self.expected = driver.run()
        self.expected.basis = HarmonicBasis([3])

    def test_from_hdf5(self):
        """Test from_hdf5."""
        with h5py.File(
            self.get_resource_path(
                "vibrational_structure_driver_result.hdf5",
                "second_q/operator_factories/vibrational/resources",
            ),
            "r",
        ) as file:
            for group in file.values():
                prop = VibrationalStructureDriverResult.from_hdf5(group)
                for inner_prop in iter(prop):
                    expected = self.expected.get_property(type(inner_prop))
                    self.assertEqual(inner_prop, expected)
