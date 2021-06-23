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

""" Test Direct Mapper """

import unittest

from test import QiskitNatureTestCase
from test.mappers.second_quantization.resources.reference_direct_mapper import REFERENCE

from qiskit_nature.drivers.second_quantization import GaussianForcesDriver
from qiskit_nature.mappers.second_quantization import DirectMapper
from qiskit_nature.properties.second_quantization.vibrational import VibrationalEnergy
from qiskit_nature.properties.second_quantization.vibrational.bases import HarmonicBasis


class TestDirectMapper(QiskitNatureTestCase):
    """Test Direct Mapper"""

    def test_mapping(self):
        """Test mapping to qubit operator"""
        driver = GaussianForcesDriver(
            logfile=self.get_resource_path(
                "CO2_freq_B3LYP_ccpVDZ.log",
                "problems/second_quantization/vibrational/resources",
            )
        )
        watson_hamiltonian = driver.run()
        num_modes = watson_hamiltonian.num_modes
        num_modals = [2] * num_modes

        vibration_energy = VibrationalEnergy.from_driver_result(watson_hamiltonian)
        vibration_energy.basis = HarmonicBasis(num_modals)

        vibration_op = vibration_energy.second_q_ops()[0]

        mapper = DirectMapper()
        qubit_op = mapper.map(vibration_op)

        self.assertEqual(qubit_op, REFERENCE)

    def test_allows_two_qubit_reduction(self):
        """Test this returns False for this mapper"""
        mapper = DirectMapper()
        self.assertFalse(mapper.allows_two_qubit_reduction)


if __name__ == "__main__":
    unittest.main()
