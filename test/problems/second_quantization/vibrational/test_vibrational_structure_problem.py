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

import warnings
from test import QiskitNatureTestCase

import numpy as np
from qiskit.opflow import PauliSumOp

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.drivers.second_quantization import GaussianForcesDriver
from qiskit_nature.mappers.second_quantization import DirectMapper
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.problems.second_quantization import \
    VibrationalStructureProblem

from .resources.expected_ops import (_num_modals_2_q_op, _num_modals_3_q_op,
                                     _truncation_order_1_op,
                                     _truncation_order_2_op)


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

    def compare_pauli_sum_op(self, first: PauliSumOp, second: PauliSumOp, msg: str = None) -> None:
        """Compares two ElectronicIntegrals instances."""
        for (f_lbl, f_coeff), (s_lbl, s_coeff) in zip(
            first.primitive.to_list(), second.primitive.to_list()
        ):
            if f_lbl != s_lbl:
                raise self.failureException(msg)
            if not np.isclose(f_coeff, s_coeff):
                raise self.failureException(msg)

    def setUp(self) -> None:
        """Setup."""
        super().setUp()
        self.addTypeEqualityFunc(VibrationalOp, self.compare_vibrational_op)
        self.addTypeEqualityFunc(PauliSumOp, self.compare_pauli_sum_op)

        logfile = self.get_resource_path(
            "test_driver_gaussian_log_C01.txt",
            "drivers/second_quantization/gaussiand",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.driver = GaussianForcesDriver(logfile=logfile)
            self.props = self.driver.run()

    def test_second_q_ops_without_transformers(self):
        """Tests that the list of second quantized operators is created if no transformers
        provided."""
        expected_num_of_sec_quant_ops = 5
        expected_len_of_vibrational_op = 47
        num_modals = 2
        truncation_order = 3
        num_modes = self.props.num_modes
        num_modals = [num_modals] * num_modes
        vibrational_problem = VibrationalStructureProblem(self.driver, num_modals, truncation_order)
        second_quantized_ops = vibrational_problem.second_q_ops()
        vibrational_op = second_quantized_ops[0]

        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check expected length of the vibrational op."):
            assert len(vibrational_op) == expected_len_of_vibrational_op
        with self.subTest("Check types in the list of second quantized operators."):
            assert isinstance(vibrational_op, VibrationalOp)
            self.assertEqual(vibrational_op, _truncation_order_2_op)

    def test_truncation_order(self):
        """Tests that the truncation_order is being respected."""
        expected_num_of_sec_quant_ops = 5
        expected_len_of_vibrational_op = 10
        num_modals = 2
        truncation_order = 1
        num_modes = self.props.num_modes
        num_modals = [num_modals] * num_modes
        vibrational_problem = VibrationalStructureProblem(self.driver, num_modals, truncation_order)
        second_quantized_ops = vibrational_problem.second_q_ops()
        vibrational_op = second_quantized_ops[0]

        with self.subTest("Check expected length of the list of second quantized operators."):
            assert len(second_quantized_ops) == expected_num_of_sec_quant_ops
        with self.subTest("Check expected length of the vibrational op."):
            assert len(vibrational_op) == expected_len_of_vibrational_op
        with self.subTest("Check types in the list of second quantized operators."):
            assert isinstance(vibrational_op, VibrationalOp)
            self.assertEqual(vibrational_op, _truncation_order_1_op)

    def test_tutorial(self):
        """Test the operators generated in the vibrational strcuture tutorial."""
        qubit_converter = QubitConverter(DirectMapper())
        truncation_order = 2

        with self.subTest("num_modals=2"):
            num_modals = 2
            num_modes = self.props.num_modes
            num_modals = [num_modals] * num_modes
            vibrational_problem = VibrationalStructureProblem(self.driver, num_modals, truncation_order)
            second_quantized_ops = vibrational_problem.second_q_ops()
            vibrational_op = second_quantized_ops[0]
            qubit_op = qubit_converter.convert(vibrational_op)
            self.assertEqual(qubit_op, _num_modals_2_q_op)

        with self.subTest("num_modals=3"):
            num_modals = 3
            num_modes = self.props.num_modes
            num_modals = [num_modals] * num_modes
            vibrational_problem = VibrationalStructureProblem(self.driver, num_modals, truncation_order)
            second_quantized_ops = vibrational_problem.second_q_ops()
            vibrational_op = second_quantized_ops[0]
            qubit_op = qubit_converter.convert(vibrational_op)
            self.assertEqual(qubit_op, _num_modals_3_q_op)
