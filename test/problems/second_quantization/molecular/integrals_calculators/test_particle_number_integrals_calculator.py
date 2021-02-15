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

from qiskit_nature.problems.second_quantization.molecular.integrals_calculators \
    .particle_number_integrals_calculator import \
    calc_total_particle_num_ints


class TestParticleNumberIntegralsCalculator(QiskitNatureTestCase):
    """Tests total particle number integrals calculator."""

    def test_calc_total_particle_num_ints(self):
        """Tests that one- and two-body integrals are calculated correctly."""
        num_modes = 1
        expected_h_1 = np.eye(num_modes, dtype=complex)
        expected_h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        h_1, h_2 = calc_total_particle_num_ints(num_modes)

        assert h_1 == expected_h_1
        assert h_2 == expected_h_2
