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

import warnings
from test import QiskitNatureTestCase
from test.problems.second_quantization.vibrational.resources.expected_labels import (
    _co2_freq_b3lyp_sparse_labels as expected_labels,
)
from test.problems.second_quantization.vibrational.resources.expected_labels import (
    _co2_freq_b3lyp_coeffs as expected_coeffs,
)

from qiskit_nature.drivers import GaussianForcesDriver
from qiskit_nature.drivers.bosonic_bases import HarmonicBasis
from qiskit_nature.problems.second_quantization.vibrational.builders.vibrational_label_builder import (
    _create_labels,
)


class TestVibrationalLabelBuilder(QiskitNatureTestCase):
    """Tests Vibrational Label Builder."""

    def test_create_labels(self):
        """Tests that correct labels are built."""
        logfile = self.get_resource_path(
            "CO2_freq_B3LYP_ccpVDZ.log",
            "problems/second_quantization/vibrational/resources",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            driver = GaussianForcesDriver(logfile=logfile)
            watson_hamiltonian = driver.run()

        num_modals = 2
        truncation_order = 3
        num_modes = watson_hamiltonian.num_modes
        num_modals = [num_modals] * num_modes

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            boson_hamilt_harm_basis = HarmonicBasis(
                watson_hamiltonian, num_modals, truncation_order
            ).convert()

        labels, coeffs = zip(*_create_labels(boson_hamilt_harm_basis))
        self.assertSetEqual(frozenset(labels), frozenset(expected_labels))
        self.assertSetEqual(frozenset(coeffs), frozenset(expected_coeffs))
