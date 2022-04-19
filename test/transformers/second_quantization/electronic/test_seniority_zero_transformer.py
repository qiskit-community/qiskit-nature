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

"""Tests for the SeniorityZeroTransformer."""

import unittest

from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.drivers.second_quantization.hdf5d.hdf5driver import HDF5Driver
from qiskit_nature.properties.second_quantization.electronic.bases.electronic_basis import (
    ElectronicBasis,
)
from qiskit_nature.transformers.second_quantization.electronic.seniority_zero_transformer import (
    SeniorityZeroTransformer,
)


class TestSeniorityZeroTransformer(QiskitNatureTestCase):
    """SeniorityZeroTransformer tests."""

    def assertDriverResult(self, driver_result, expected):
        """Asserts that the two `DriverResult` object's relevant fields are equivalent."""
        electronic_energy = driver_result.get_property("ElectronicEnergy")
        electronic_energy_exp = expected.get_property("ElectronicEnergy")
        with self.subTest("MO 1-electron integrals"):
            np.testing.assert_array_almost_equal(
                electronic_energy.get_electronic_integral(ElectronicBasis.MO, 1).get_matrix(),
                electronic_energy_exp.get_electronic_integral(ElectronicBasis.MO, 1).get_matrix(),
            )
        with self.subTest("MO 2-electron integrals"):
            np.testing.assert_array_almost_equal(
                electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2).get_matrix(),
                electronic_energy_exp.get_electronic_integral(ElectronicBasis.MO, 2).get_matrix(),
            )

    def test_transformation_to_restricted_operator(self):
        """Test transformation to restricted operator."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "LiH_sto3g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        driver_result = driver.run()

        transformer = SeniorityZeroTransformer()
        driver_result_restricted = transformer.transform(driver_result)

        expected = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "LiH_sto3g_restricted.hdf5", "transformers/second_quantization/electronic"
            )
        ).run()

        self.assertDriverResult(driver_result_restricted, expected)


if __name__ == "__main__":
    unittest.main()
