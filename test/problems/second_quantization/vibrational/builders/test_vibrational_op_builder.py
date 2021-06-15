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
"""Tests Vibrational Operator builder."""

from test import QiskitNatureTestCase
from test.problems.second_quantization.vibrational.resources.expected_labels import (
    _co2_freq_b3lyp_dense_labels as expected_labels,
)
from test.problems.second_quantization.vibrational.resources.expected_labels import (
    _co2_freq_b3lyp_coeffs as expected_coeffs,
)

from qiskit_nature.operators.second_quantization.vibrational_op import VibrationalOp
from qiskit_nature.problems.second_quantization.vibrational.builders.vibrational_op_builder import (
    _build_vibrational_op,
)
from qiskit_nature.drivers.second_quantization import GaussianForcesDriver


class TestVibrationalOpBuilder(QiskitNatureTestCase):
    """Tests Vibrational Op builder."""

    def test_vibrational_op_builder(self):
        """Tests that a VibrationalOp is created correctly from a driver."""
        logfile = self.get_resource_path(
            "CO2_freq_B3LYP_ccpVDZ.log",
            "problems/second_quantization/vibrational/resources",
        )
        driver = GaussianForcesDriver(logfile=logfile)

        watson_hamiltonian = driver.run()
        num_modals = 2
        truncation_order = 3

        vibrational_op = _build_vibrational_op(watson_hamiltonian, num_modals, truncation_order)

        assert isinstance(vibrational_op, VibrationalOp)
        labels, coeffs = zip(*vibrational_op.to_list())
        self.assertSetEqual(frozenset(labels), frozenset(expected_labels))
        self.assertSetEqual(frozenset(coeffs), frozenset(expected_coeffs))
