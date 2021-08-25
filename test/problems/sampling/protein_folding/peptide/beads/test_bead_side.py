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
"""Tests Side Bead."""
from test import QiskitNatureTestCase
from test.problems.sampling.protein_folding.resources.file_parser import read_expected_file
from qiskit.opflow import I, Z

from qiskit_nature.problems.sampling.protein_folding.peptide.pauli_ops_builder import (
    _build_full_identity,
)
from qiskit_nature.problems.sampling.protein_folding.peptide.beads.side_bead import SideBead

PATH = "problems/sampling/protein_folding/resources/test_bead_side"


class TestSideBead(QiskitNatureTestCase):
    """Tests Side Bead."""

    def test_side_bead_constructor(self):
        """Tests that a SideBead is created."""
        residue_type = "S"
        main_chain_len = 4
        num_turn_qubits = 2 * (main_chain_len - 1)
        main_chain_id = 3
        side_bead_id = 3
        turn_qubits = (
            0.5 * _build_full_identity(num_turn_qubits) - 0.5 * (I ^ I ^ I ^ I ^ I ^ Z),
            0.5 * _build_full_identity(num_turn_qubits) - 0.5 * (I ^ I ^ I ^ I ^ Z ^ I),
        )
        side_bead = SideBead(main_chain_id, side_bead_id, residue_type, turn_qubits)

        indic_0, indic_1, indic_2, indic_3 = side_bead.indicator_functions
        expected_path_indic_0 = self.get_resource_path(
            "test_side_bead_constructor_expected_indic_0",
            PATH,
        )
        expected_indic_0 = read_expected_file(expected_path_indic_0)

        expected_path_indic_1 = self.get_resource_path(
            "test_side_bead_constructor_expected_indic_1",
            PATH,
        )
        expected_indic_1 = read_expected_file(expected_path_indic_1)

        expected_path_indic_2 = self.get_resource_path(
            "test_side_bead_constructor_expected_indic_2",
            PATH,
        )
        expected_indic_2 = read_expected_file(expected_path_indic_2)

        expected_path_indic_3 = self.get_resource_path(
            "test_side_bead_constructor_expected_indic_3",
            PATH,
        )
        expected_indic_3 = read_expected_file(expected_path_indic_3)

        self.assertEqual(indic_0, expected_indic_0)
        self.assertEqual(indic_1, expected_indic_1)
        self.assertEqual(indic_2, expected_indic_2)
        self.assertEqual(indic_3, expected_indic_3)

    def test_side_bead_constructor_none(self):
        """Tests that a SideBead is created."""
        residue_type = ""
        turn_qubits = (Z, Z)
        main_chain_id = 3
        side_bead_id = 3
        side_bead = SideBead(main_chain_id, side_bead_id, residue_type, turn_qubits)

        with self.assertRaises(AttributeError):
            _, _, _, _ = side_bead.indicator_functions
