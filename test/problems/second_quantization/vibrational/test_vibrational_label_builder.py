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
"""Tests Vibrational Label Builder."""
from test import QiskitNatureTestCase
from test.problems.second_quantization.vibrational.resources.expected_labels import \
    _co2_freq_b3lyp_labels
from qiskit_nature.drivers import GaussianForcesDriver
from qiskit_nature.components.bosonic_bases import HarmonicBasis
from qiskit_nature.problems.second_quantization.vibrational.vibrational_label_builder import \
    _create_labels


class TestVibrationalLabelBuilder(QiskitNatureTestCase):
    """Tests Vibrational Label Builder."""

    def test_create_labels(self):
        """Tests that correct labels are built."""
        expected_labels = _co2_freq_b3lyp_labels
        logfile = self.get_resource_path('CO2_freq_B3LYP_ccpVDZ.log')
        driver = GaussianForcesDriver(logfile=logfile)
        watson_hamiltonian = driver.run()
        basis_size = 2
        truncation_order = 3
        num_modes = watson_hamiltonian.num_modes
        basis_size = [basis_size] * num_modes
        boson_hamilt_harm_basis = HarmonicBasis(watson_hamiltonian,  # type: ignore
                                                basis_size, truncation_order).convert()
        labels = _create_labels(boson_hamilt_harm_basis)
        assert labels == expected_labels
