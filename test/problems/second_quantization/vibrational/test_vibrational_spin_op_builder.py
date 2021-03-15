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
"""Tests Fermionic Operator builder."""
from test import QiskitNatureTestCase
from qiskit_nature.operators.second_quantization.vibrational_spin_op import VibrationalSpinOp
from qiskit_nature.problems.second_quantization.vibrational.vibrational_spin_op_builder import \
    build_vibrational_spin_op
from qiskit_nature.drivers import GaussianForcesDriver


class TestVibrationalSpinOpBuilder(QiskitNatureTestCase):
    """Tests Vibrational Spin Op builder."""

    def test_vibrational_spin_op_builder(self):
        """Tests that a VibrationalSpinOp is created correctly from a driver."""
        logfile = self.get_resource_path('CO2_freq_B3LYP_ccpVDZ.log')
        driver = GaussianForcesDriver(logfile=logfile)

        watson_hamiltonian = driver.run()
        basis_size = 2
        truncation_order = 3

        vibrational_spin_op = build_vibrational_spin_op(watson_hamiltonian, basis_size,
                                                        truncation_order)

        assert isinstance(vibrational_spin_op, VibrationalSpinOp)
