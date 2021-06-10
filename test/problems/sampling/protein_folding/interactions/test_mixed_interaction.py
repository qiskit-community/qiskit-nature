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
from problems.sampling.protein_folding.exceptions.invalid_residue_exception import \
    InvalidResidueException
from qiskit_nature.problems.sampling.protein_folding.interactions.mixed_interaction import \
    MixedInteraction
from test import QiskitNatureTestCase


class TestMixedInteraction(QiskitNatureTestCase):
    """Tests MixedInteraction."""

    def test_calc_energy_matrix(self):
        """Tests that energy matrix is calculated correctly."""
        additional_energies = None
        interaction = MixedInteraction(additional_energies)
        num_beads = 3
        sequence = ["A","A","S"]
        energy_matrix = interaction.calc_energy_matrix(num_beads, sequence)

        assert len(energy_matrix) == num_beads + 1
        assert len(energy_matrix[0]) == 2

    def test_calc_energy_matrix_invalid_residue(self):
        """Tests that an exception is thrown when an invalid residue is provided."""
        interaction = MixedInteraction()
        num_beads = 1
        sequence = ["Z"]

        with self.assertRaises(InvalidResidueException):
            _ = interaction.calc_energy_matrix(num_beads, sequence)
