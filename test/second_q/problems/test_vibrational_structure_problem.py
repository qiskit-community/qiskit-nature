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

"""Tests Vibrational Problem."""

import unittest
from test import QiskitNatureTestCase
from test.second_q.problems.resources.expected_ops import (
    _truncation_order_1_op,
    _truncation_order_2_op,
)

from qiskit_nature.second_q.drivers import GaussianForcesDriver
from qiskit_nature.second_q.operators import VibrationalOp
from qiskit_nature.second_q.problems import HarmonicBasis
import qiskit_nature.optionals as _optionals


class TestVibrationalStructureProblem(QiskitNatureTestCase):
    """Tests Vibrational Problem."""

    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        logfile = self.get_resource_path(
            "test_driver_gaussian_log_C01.txt",
            "second_q/drivers/gaussiand",
        )
        self.driver = GaussianForcesDriver(logfile=logfile)
        self.basis = HarmonicBasis([2, 2, 2, 2])
        self.problem = self.driver.run(basis=self.basis)

    def test_second_q_ops_without_transformers(self):
        """Tests that the list of second quantized operators is created if no transformers
        provided."""
        expected_num_of_sec_quant_ops = 4
        expected_len_of_vibrational_op = 46
        truncation_order = 3
        vibrational_problem = self.problem
        vibrational_problem.hamiltonian.truncation_order = truncation_order
        vibrational_op, second_quantized_ops = vibrational_problem.second_q_ops()
        vibrational_op = vibrational_op.normal_order()

        with self.subTest("Check is an instance of VibrationalOp."):
            self.assertIsInstance(vibrational_op, VibrationalOp)
        with self.subTest("Check expected length of the list of second quantized operators."):
            self.assertEqual(len(second_quantized_ops), expected_num_of_sec_quant_ops)
        with self.subTest("Check expected length of the vibrational op."):
            self.assertEqual(len(vibrational_op), expected_len_of_vibrational_op)
        with self.subTest("Check types in the list of second quantized operators."):
            self.assertTrue(vibrational_op.equiv(_truncation_order_2_op))

    def test_truncation_order(self):
        """Tests that the truncation_order is being respected."""
        expected_num_of_sec_quant_ops = 4
        expected_len_of_vibrational_op = 10
        truncation_order = 1
        vibrational_problem = self.problem
        vibrational_problem.hamiltonian.truncation_order = truncation_order
        vibrational_op, second_quantized_ops = vibrational_problem.second_q_ops()
        vibrational_op = vibrational_op.normal_order()

        with self.subTest("Check is an instance of VibrationalOp."):
            self.assertIsInstance(vibrational_op, VibrationalOp)
        with self.subTest("Check expected length of the list of second quantized operators."):
            self.assertEqual(len(second_quantized_ops), expected_num_of_sec_quant_ops)
        with self.subTest("Check expected length of the vibrational op."):
            self.assertEqual(len(vibrational_op), expected_len_of_vibrational_op)
        with self.subTest("Check types in the list of second quantized operators."):
            self.assertTrue(vibrational_op.equiv(_truncation_order_1_op))


if __name__ == "__main__":
    unittest.main()
