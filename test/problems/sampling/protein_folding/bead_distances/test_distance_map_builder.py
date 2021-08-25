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
from qiskit_nature.problems.sampling.protein_folding.bead_distances.distance_map_builder import (
    DistanceMapBuilder,
)
from qiskit_nature.problems.sampling.protein_folding.bead_distances.distance_map import DistanceMap
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide

PATH = "problems/sampling/protein_folding/resources/test_distance_map_builder"


class TestDistanceMapBuilder(QiskitNatureTestCase):
    """Tests DistanceMapBuilder."""

    def setUp(self):
        super().setUp()
        main_chain_residue_seq = "SAAAA"
        side_chain_residue_sequences = ["", "", "A", "", ""]

        self.peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)
        self.distance_map_builder = DistanceMapBuilder()

    def test_calc_distances_main_chain(self):
        """
        Tests that distances for all beads on the main chain are calculated correctly.
        """
        self.distance_map_builder._calc_distances_main_chain(self.peptide)

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

        lower_main_bead_0 = self.peptide.get_main_chain[0]
        upper_main_bead_4 = self.peptide.get_main_chain[4]

        lower_main_bead_1 = self.peptide.get_main_chain[1]
        upper_main_bead_3 = self.peptide.get_main_chain[3]

        # checking only some of the entries
        self.assertEqual(
            self.distance_map_builder._distance_map_axes[0][lower_main_bead_0][upper_main_bead_4],
            expected_1,
        )
        self.assertEqual(
            self.distance_map_builder._distance_map_axes[0][lower_main_bead_1][upper_main_bead_4],
            expected_2,
        )
        self.assertEqual(
            self.distance_map_builder._distance_map_axes[1][lower_main_bead_0][upper_main_bead_3],
            expected_3,
        )
        self.assertEqual(
            self.distance_map_builder._distance_map_axes[1][lower_main_bead_0][upper_main_bead_4],
            expected_4,
        )

    def test_add_distances_side_chain(self):
        """
        Tests that distances for all beads on side chains are calculated correctly.
        """
        self.distance_map_builder._calc_distances_main_chain(self.peptide)
        self.distance_map_builder._add_distances_side_chain(self.peptide)
        expected_path = self.get_resource_path(
            "test_add_distances_side_chain",
            PATH,
        )
        expected = read_expected_file(expected_path)
        lower_main_bead = self.peptide.get_main_chain[0]
        upper_side_bead = self.peptide.get_main_chain[2].side_chain[0]
        # checking only some of the entries
        self.assertEqual(
            self.distance_map_builder._distance_map_axes[0][lower_main_bead][upper_side_bead],
            expected,
        )

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
