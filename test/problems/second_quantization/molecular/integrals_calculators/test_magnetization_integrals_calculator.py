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

"""Tests total magnetization integrals calculator."""
from test import QiskitNatureTestCase
import numpy as np

from qiskit_nature.problems.second_quantization.molecular.integrals_calculators import \
    calc_total_magnetization_ints


# TODO add more detailed tests
class TestMolecularProblem(QiskitNatureTestCase):
    """Tests total magnetization integrals calculator."""

    def test_calc_total_magnetization_ints(self):
        """Tests that one-body integrals for total magnetization are calculated correctly."""
        num_modes = 1
        expected_h_1 = [[-0.5 + 0.j]]
        h_1 = calc_total_magnetization_ints(num_modes)

        assert np.allclose(h_1, expected_h_1)
