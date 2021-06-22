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
"""Tests DistanceMapBuilder."""
from test import QiskitNatureTestCase
from test.problems.sampling.protein_folding.resources.file_parser import read_expected_file
from qiskit_nature.problems.sampling.protein_folding.bead_distances import distance_map_builder
from qiskit_nature.problems.sampling.protein_folding.bead_distances.distance_map import DistanceMap
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide

PATH = "problems/sampling/protein_folding/resources/test_distance_map_builder"


class TestDistanceMapBuilder(QiskitNatureTestCase):
    """Tests DistanceMapBuilder."""

    def setUp(self):
        super().setUp()
        main_chain_residue_seq = ["S", "A", "A", "A", "A"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 0, 0]
        side_chain_residue_sequences = [None, None, "A", None, None]

        self.peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )

    def test_calc_distances_main_chain(self):
        """
        Tests that distances for all beads on the main chain are calculated correctly.
        """
        delta_n0, delta_n1, _, _ = distance_map_builder._calc_distances_main_chain(self.peptide)

        expected_path = self.get_resource_path(
            "test_calc_distances_main_chain_1",
            PATH,
        )
        expected_1 = read_expected_file(expected_path)
        expected_path = self.get_resource_path(
            "test_calc_distances_main_chain_2",
            PATH,
        )

        expected_2 = read_expected_file(expected_path)
        expected_path = self.get_resource_path(
            "test_calc_distances_main_chain_3",
            PATH,
        )

        expected_3 = read_expected_file(expected_path)
        expected_path = self.get_resource_path(
            "test_calc_distances_main_chain_4",
            PATH,
        )

        expected_4 = read_expected_file(expected_path)
        # checking only some of the entries
        self.assertEqual(delta_n0[1][0][5][0], expected_1)
        self.assertEqual(delta_n0[2][0][5][0], expected_2)
        self.assertEqual(delta_n1[1][0][4][0], expected_3)
        self.assertEqual(delta_n1[1][0][5][0], expected_4)

    def test_add_distances_side_chain(self):
        """
        Tests that distances for all beads on side chains are calculated correctly.
        """
        (
            delta_n0_main,
            delta_n1_main,
            delta_n2_main,
            delta_n3_main,
        ) = distance_map_builder._calc_distances_main_chain(self.peptide)
        delta_n0, _, _, _ = distance_map_builder._add_distances_side_chain(
            self.peptide, delta_n0_main, delta_n1_main, delta_n2_main, delta_n3_main
        )
        expected_path = self.get_resource_path(
            "test_add_distances_side_chain",
            PATH,
        )
        expected = read_expected_file(expected_path)
        # checking only some of the entries
        self.assertEqual(delta_n0[1][0][3][1], expected)

    def test_calc_total_distances(self):
        """
        Tests that total distances for all beads are calculated correctly.
        """
        distance_map = DistanceMap(self.peptide)
        upper_bead_1 = self.peptide.get_main_chain[2].side_chain[0]
        lower_bead_1 = self.peptide.get_main_chain[1]

        upper_bead_2 = self.peptide.get_main_chain[2].side_chain[0]
        lower_bead_2 = self.peptide.get_main_chain[0]

        expected_path = self.get_resource_path(
            "test_calc_total_distances_1",
            PATH,
        )
        expected_1 = read_expected_file(expected_path)
        expected_path = self.get_resource_path(
            "test_calc_total_distances_2",
            PATH,
        )

        expected_2 = read_expected_file(expected_path)

        self.assertEqual(distance_map[lower_bead_1, upper_bead_1], expected_1)
        self.assertEqual(distance_map[lower_bead_2, upper_bead_2], expected_2)
