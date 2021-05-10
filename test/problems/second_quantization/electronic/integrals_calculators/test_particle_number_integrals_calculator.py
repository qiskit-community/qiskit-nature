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
from ddt import ddt, data

import numpy as np

from qiskit_nature.problems.second_quantization.electronic.integrals_calculators import (
    calc_total_particle_num_ints,
)


@ddt
class TestParticleNumberIntegralsCalculator(QiskitNatureTestCase):
    """Tests total particle number integrals calculator."""

    num_modes_list = [1, 2, 3]
    expected_h_1_list = [
        np.eye(1, dtype=complex),
        np.eye(2, dtype=complex),
        np.eye(3, dtype=complex),
    ]
    expected_h_2_list = [None, None, None]

    @data(*num_modes_list)
    def test_calc_total_particle_num_ints(self, num_modes):
        """Tests that one-body integrals for total particle number are calculated correctly."""

        expected_h_1 = self.expected_h_1_list[num_modes - 1]
        expected_h_2 = self.expected_h_2_list[num_modes - 1]

        h_1, h_2 = calc_total_particle_num_ints(num_modes)
        assert np.allclose(h_1, expected_h_1)
        assert h_2 == expected_h_2
