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
"""Tests DistanceMap."""
from test import QiskitNatureTestCase
from test.problems.sampling.protein_folding.resources.file_parser import read_expected_file
from qiskit_nature.problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)

from qiskit_nature.problems.sampling.protein_folding.bead_distances.distance_map import DistanceMap
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide

PATH = "problems/sampling/protein_folding/resources/test_distance_map"


class TestDistanceMap(QiskitNatureTestCase):
    """Tests DistanceMap."""

    def test_first_neighbor(self):
        """
        Tests that Pauli operators for 1st neighbor interactions are created correctly.
        """

        main_chain_residue_seq = "SAASSS"
        side_chain_residue_sequences = ["", "", "A", "S", "S", ""]

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calculate_energy_matrix(main_chain_residue_seq)
        lambda_1 = 2
        lower_main_bead_index = 1
        upper_main_bead_index = 4
        side_chain_lower_main_bead = 0
        side_chain_upper_main_bead = 0
        distance_map = DistanceMap(peptide)
        expr = distance_map._first_neighbor(
            peptide,
            lower_main_bead_index,
            side_chain_lower_main_bead,
            upper_main_bead_index,
            side_chain_upper_main_bead,
            lambda_1,
            pair_energies,
        )

        expected_path = self.get_resource_path(
            "test_first_neighbor",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(expr, expected)

    def test_first_neighbor_side(self):
        """
        Tests that Pauli operators for 1st neighbor interactions are created correctly.
        """

        main_chain_residue_seq = "SAASSS"
        side_chain_residue_sequences = ["", "", "A", "S", "S", ""]

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calculate_energy_matrix(main_chain_residue_seq)
        lambda_1 = 2
        lower_main_bead_index = 3
        upper_main_bead_index = 5
        side_chain_lower_main_bead = 1
        side_chain_upper_main_bead = 1
        distance_map = DistanceMap(peptide)
        expr = distance_map._first_neighbor(
            peptide,
            lower_main_bead_index,
            side_chain_lower_main_bead,
            upper_main_bead_index,
            side_chain_upper_main_bead,
            lambda_1,
            pair_energies,
        )

        expected_path = self.get_resource_path(
            "test_first_neighbor_side",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(expr, expected)

    def test_second_neighbor(self):
        """
        Tests that Pauli operators for 2nd neighbor interactions are created correctly.
        """
        main_chain_residue_seq = "SAASS"
        side_chain_residue_sequences = ["", "", "A", "", ""]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calculate_energy_matrix(main_chain_residue_seq)

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)
        lambda_1 = 2
        lower_main_bead_index = 1
        upper_main_bead_index = 4
        side_chain_lower_main_bead = 0
        side_chain_upper_main_bead = 0
        distance_map = DistanceMap(peptide)
        second_neighbor = distance_map._second_neighbor(
            peptide,
            lower_main_bead_index,
            side_chain_upper_main_bead,
            upper_main_bead_index,
            side_chain_lower_main_bead,
            lambda_1,
            pair_energies,
        )
        expected_path = self.get_resource_path(
            "test_second_neighbor",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(second_neighbor, expected)

    def test_second_neighbor_2(self):
        """
        Tests that Pauli operators for 2nd neighbor interactions are created correctly.
        """
        main_chain_residue_seq = "SAACS"
        side_chain_residue_sequences = ["", "", "A", "A", ""]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calculate_energy_matrix(main_chain_residue_seq)

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)
        lambda_1 = 2
        lower_main_bead_index = 3
        upper_main_bead_index = 4
        side_chain_lower_main_bead = 1
        side_chain_upper_main_bead = 1
        distance_map = DistanceMap(peptide)
        second_neighbor = distance_map._second_neighbor(
            peptide,
            lower_main_bead_index,
            side_chain_upper_main_bead,
            upper_main_bead_index,
            side_chain_lower_main_bead,
            lambda_1,
            pair_energies,
        )
        expected_path = self.get_resource_path(
            "test_second_neighbor_2",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(second_neighbor, expected)
