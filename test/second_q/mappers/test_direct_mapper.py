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

""" Test Direct Mapper """

import unittest

from test import QiskitNatureTestCase
from test.second_q.mappers.resources.reference_direct_mapper import (
    _num_modals_2_q_op,
    _num_modals_3_q_op,
)

from qiskit_nature.second_q.drivers import GaussianForcesDriver
from qiskit_nature.second_q.mappers import DirectMapper
from qiskit_nature.second_q.problems import HarmonicBasis
import qiskit_nature.optionals as _optionals


class TestDirectMapper(QiskitNatureTestCase):
    """Test Direct Mapper"""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        self.driver = GaussianForcesDriver(
            logfile=self.get_resource_path(
                "test_driver_gaussian_log_C01.txt",
                "second_q/drivers/gaussiand",
            )
        )

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def test_mapping(self):
        """Test mapping to qubit operator"""
        num_modals = [2, 2, 2, 2]
        basis = HarmonicBasis(num_modals)
        problem = self.driver.run(basis)

        vibration_energy = problem.hamiltonian
        vibration_op = vibration_energy.second_q_op()

        mapper = DirectMapper()
        qubit_op = mapper.map(vibration_op)

        self.assertEqual(qubit_op, _num_modals_2_q_op)

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def test_larger_tutorial_qubit_op(self):
        """Test the 3-modal qubit operator generated in the vibrational structure tutorial."""
        num_modals = [3, 3, 3, 3]
        basis = HarmonicBasis(num_modals)
        problem = self.driver.run(basis)

        vibration_energy = problem.hamiltonian
        vibration_op = vibration_energy.second_q_op()

        mapper = DirectMapper()
        qubit_op = mapper.map(vibration_op)

        self.assertEqual(qubit_op, _num_modals_3_q_op)

    def test_allows_two_qubit_reduction(self):
        """Test this returns False for this mapper"""
        mapper = DirectMapper()
        self.assertFalse(mapper.allows_two_qubit_reduction)


if __name__ == "__main__":
    unittest.main()
