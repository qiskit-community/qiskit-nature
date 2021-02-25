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

"""Tests total particle number integrals calculator."""
from test import QiskitNatureTestCase
import numpy as np

from qiskit_nature.problems.second_quantization.molecular.integrals_calculators import \
    calc_total_particle_num_ints


class TestParticleNumberIntegralsCalculator(QiskitNatureTestCase):
    """Tests total particle number integrals calculator."""

    def test_calc_total_particle_num_ints(self):
        """Tests that one-body integrals for total particle number are calculated correctly."""
        num_modes = 1
        expected_h_1 = np.eye(num_modes, dtype=complex)

        h_1 = calc_total_particle_num_ints(num_modes)

        assert np.allclose(h_1, expected_h_1)
