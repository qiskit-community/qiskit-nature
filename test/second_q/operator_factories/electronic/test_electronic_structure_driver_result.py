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

"""Test ElectronicStructureDriverResult Property"""

from test.second_q.operator_factories.property_test import PropertyTest

import h5py

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.second_q.operator_factories.electronic import (
    ElectronicStructureDriverResult,
)


class TestElectronicStructureDriverResult(PropertyTest):
    """Test ElectronicStructureDriverResult Property"""

    def setUp(self) -> None:
        """Setup expected object."""
        super().setUp()

        driver = HDF5Driver(
            self.get_resource_path("BeH_sto3g_reduced.hdf5", "transformers/second_q/electronic")
        )
        self.expected = driver.run()

    def test_from_hdf5(self):
        """Test from_hdf5."""
        with h5py.File(
            self.get_resource_path(
                "electronic_structure_driver_result.hdf5",
                "properties/second_q/electronic/resources",
            ),
            "r",
        ) as file:
            for group in file.values():
                prop = ElectronicStructureDriverResult.from_hdf5(group)
                for inner_prop in iter(prop):
                    expected = self.expected.get_property(type(inner_prop))
                    self.assertEqual(inner_prop, expected)
