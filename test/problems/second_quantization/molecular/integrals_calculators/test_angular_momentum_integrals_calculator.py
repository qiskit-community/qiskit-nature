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

"""Tests total angular momentum integrals calculator."""
from test import QiskitNatureTestCase
import numpy as np

from qiskit_nature.problems.second_quantization.molecular.integrals_calculators \
    .angular_momentum_integrals_calculator import \
    calc_total_ang_momentum_ints


# TODO add more detailed tests
class TestMolecularProblem(QiskitNatureTestCase):
    """Tests total angular momentum integrals calculator."""

    def test_calc_total_ang_momentum_ints(self):
        """Tests that one- and two-body integrals for total angular momentum are calculated
        correctly."""
        num_modes = 1

        h_1, h_2 = calc_total_ang_momentum_ints(num_modes)

        assert isinstance(h_1, np.ndarray)
        assert isinstance(h_2, np.ndarray)
