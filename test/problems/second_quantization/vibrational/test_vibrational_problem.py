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

"""Tests Vibrational Problem."""
from test import QiskitNatureTestCase
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.problems.second_quantization import VibrationalStructureProblem
from qiskit_nature.drivers import GaussianForcesDriver


class TestVibrationalProblem(QiskitNatureTestCase):
    """Tests Vibrational Problem."""

    def test_second_q_ops_without_transformers(self):
        """Tests that the list of second quantized operators is created if no transformers
        provided."""
        expected_num_of_sec_quant_ops = 5
        logfile = self.get_resource_path(
            'CO2_freq_B3LYP_ccpVDZ.log', 'problems/second_quantization/vibrational/resources'
        )
        driver = GaussianForcesDriver(logfile=logfile)

        watson_hamiltonian = driver.run()
        num_modals = 2
        truncation_order = 3
        num_modes = watson_hamiltonian.num_modes
        num_modals = [num_modals] * num_modes
        vibrational_problem = VibrationalStructureProblem(driver, num_modals, truncation_order)
        second_quantized_ops = vibrational_problem.second_q_ops()
        vibrational_op = second_quantized_ops.main_operator
        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check types in the list of second quantized operators."):
            assert isinstance(vibrational_op, VibrationalOp)
        # TODO: add more checks once the algorithms are fully in place
