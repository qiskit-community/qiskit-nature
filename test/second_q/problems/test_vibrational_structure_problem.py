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

import numpy as np

from qiskit_nature.second_q.drivers import GaussianForcesDriver
from qiskit_nature.second_q.operators import VibrationalOp
from qiskit_nature.second_q.problems import VibrationalStructureProblem

from .resources.expected_ops import _truncation_order_1_op, _truncation_order_2_op


class TestVibrationalStructureProblem(QiskitNatureTestCase):
    """Tests Vibrational Problem."""

    def compare_vibrational_op(
        self, first: VibrationalOp, second: VibrationalOp, msg: str = None
    ) -> None:
        """Compares two ElectronicIntegrals instances."""
        for f_lbl, s_lbl in zip(first._labels, second._labels):
            if f_lbl != s_lbl:
                raise self.failureException(msg)
        for f_coeff, s_coeff in zip(first._coeffs, second._coeffs):
            if not np.isclose(f_coeff, s_coeff):
                raise self.failureException(msg)

    def setUp(self) -> None:
        """Setup."""
        super().setUp()
        self.addTypeEqualityFunc(VibrationalOp, self.compare_vibrational_op)

        logfile = self.get_resource_path(
            "test_driver_gaussian_log_C01.txt",
            "second_q/drivers/gaussiand",
        )
        self.driver = GaussianForcesDriver(logfile=logfile)
        self.props = self.driver.run()

    def test_second_q_ops_without_transformers(self):
        """Tests that the list of second quantized operators is created if no transformers
        provided."""
        expected_num_of_sec_quant_ops = 4
        expected_len_of_vibrational_op = 47
        num_modals = 2
        truncation_order = 3
        num_modes = self.props.num_modes
        num_modals = [num_modals] * num_modes
        vibrational_problem = self.props
        vibrational_problem._num_modals = num_modals
        vibrational_problem.truncation_order = truncation_order
        vibrational_op, second_quantized_ops = vibrational_problem.second_q_ops()

        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check expected length of the vibrational op."):
            assert len(vibrational_op) == expected_len_of_vibrational_op
        with self.subTest("Check types in the list of second quantized operators."):
            assert isinstance(vibrational_op, VibrationalOp)
            self.assertEqual(vibrational_op, _truncation_order_2_op)

    def test_truncation_order(self):
        """Tests that the truncation_order is being respected."""
        expected_num_of_sec_quant_ops = 4
        expected_len_of_vibrational_op = 10
        num_modals = 2
        truncation_order = 1
        num_modes = self.props.num_modes
        num_modals = [num_modals] * num_modes
        vibrational_problem = self.props
        vibrational_problem._num_modals = num_modals
        vibrational_problem.truncation_order = truncation_order
        vibrational_op, second_quantized_ops = vibrational_problem.second_q_ops()

        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check expected length of the vibrational op."):
            assert len(vibrational_op) == expected_len_of_vibrational_op
        with self.subTest("Check types in the list of second quantized operators."):
            assert isinstance(vibrational_op, VibrationalOp)
            self.assertEqual(vibrational_op, _truncation_order_1_op)


if __name__ == "__main__":
    unittest.main()
