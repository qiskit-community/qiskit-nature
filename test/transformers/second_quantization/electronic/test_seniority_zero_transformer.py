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

import json
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

    def assertOperatorsEqual(self, operator, expected_operator_file):
        """Compare two operators, where the expected operator is given in a JSON file."""
        with open(
            self.get_resource_path(
                expected_operator_file, "transformers/second_quantization/electronic"
            ),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)
        for op, expected_op in zip(operator.to_list(), expected):
            self.assertEqual(op[0], expected_op[0])
            self.assertTrue(np.isclose(op[1], expected_op[1]))

    def test_transformation_to_restricted_formalism(self):
        """Test transformation to restricted formalism."""
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

    def test_restricted_second_q_ops(self):
        """Test building of second quantization operators in restricted formalism."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "LiH_sto3g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        driver_result = driver.run()

        transformer = SeniorityZeroTransformer()
        driver_result_restricted = transformer.transform(driver_result)
        second_q_ops = driver_result_restricted.second_q_ops()

        with self.subTest("Electronic Energy"):
            operator = second_q_ops["ElectronicEnergy"]
            self.assertOperatorsEqual(
                operator, expected_operator_file="LiH_sto3g_restricted_electronic_energy_op.json"
            )

        with self.subTest("Particle Number"):
            operator = second_q_ops["ParticleNumber"]
            self.assertOperatorsEqual(
                operator, expected_operator_file="LiH_sto3g_restricted_particle_number_op.json"
            )


if __name__ == "__main__":
    unittest.main()
