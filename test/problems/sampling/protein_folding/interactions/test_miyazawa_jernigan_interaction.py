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
"""Tests MiyazawaJerniganInteraction."""
from test import QiskitNatureTestCase
from qiskit_nature.problems.sampling.protein_folding.exceptions.invalid_residue_exception import (
    InvalidResidueException,
)
from qiskit_nature.problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)


class TestMiyazawaJerniganInteraction(QiskitNatureTestCase):
    """Tests MiyazawaJerniganInteraction."""

    def test_calc_energy_matrix(self):
        """Tests that energy matrix is calculated correctly."""
        interaction = MiyazawaJerniganInteraction()
        sequence = ["A", "A", "A"]
        energy_matrix = interaction.calculate_energy_matrix(sequence)

        self.assertEqual(len(energy_matrix), len(sequence) + 1)
        self.assertEqual(len(energy_matrix[0]), 2)

    def test_calc_energy_matrix_invalid_residue(self):
        """Tests that an exception is thrown when an invalid residue is provided."""
        interaction = MiyazawaJerniganInteraction()
        sequence = ["A", "A", "B"]

        with self.assertRaises(InvalidResidueException):
            _ = interaction.calculate_energy_matrix(sequence)
