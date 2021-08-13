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
"""Tests RandomInteraction."""
from test import QiskitNatureTestCase
from qiskit_nature.problems.sampling.protein_folding.interactions.random_interaction import (
    RandomInteraction,
)


class TestRandomInteraction(QiskitNatureTestCase):
    """Tests RandomInteraction."""

    def test_calc_energy_matrix(self):
        """Tests that energy matrix is calculated correctly."""
        interaction = RandomInteraction()
        sequence = ["", "", ""]
        energy_matrix = interaction.calculate_energy_matrix(sequence)

        self.assertEqual(len(energy_matrix), len(sequence) + 1)
        self.assertEqual(len(energy_matrix[0]), 2)
