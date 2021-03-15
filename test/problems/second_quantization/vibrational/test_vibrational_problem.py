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
import unittest

from test import QiskitNatureTestCase
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization.vibrational.vibrational_problem import \
    VibrationalProblem
from qiskit_nature.drivers import GaussianForcesDriver


class TestVibrationalProblem(QiskitNatureTestCase):
    """Tests Vibrational Problem."""

    # TODO complete unit test once implementation ready
    @unittest.skip("Skip test until implementation completed.")
    def test_second_q_ops_without_transformers(self):
        """Tests that the list of second quantized operators is created if no transformers
        provided."""
        expected_num_of_sec_quant_ops = 7
        logfile = self.get_resource_path('CO2_freq_B3LYP_ccpVDZ.log')
        driver = GaussianForcesDriver(logfile=logfile)

        watson_hamiltonian = driver.run()
        basis_size = 2
        truncation_order = 3
        num_modes = watson_hamiltonian.num_modes
        basis_size = [basis_size] * num_modes
        vibrational_problem = VibrationalProblem(driver, basis_size, truncation_order)
        second_quantized_ops = vibrational_problem.second_q_ops()
        electr_sec_quant_op = second_quantized_ops[0]
        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check types in the list of second quantized operators."):
            for second_quantized_op in second_quantized_ops:
                assert isinstance(second_quantized_op, SecondQuantizedOp)
        # with self.subTest("Check components of electronic second quantized operator."):
        #     assert all(s[0] == t[0] and np.isclose(s[1], t[1]) for s, t in
        #                zip(expected_spin_op, electr_sec_quant_op.spin.to_list()))
        #     assert electr_sec_quant_op.boson is None
        #     assert electr_sec_quant_op.fermion is None
