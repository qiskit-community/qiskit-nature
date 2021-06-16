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
"""Tests Main Bead."""
from test import QiskitNatureTestCase
from qiskit.opflow import I, Z

from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_full_identity
from qiskit_nature.problems.sampling.protein_folding.peptide.beads.main_bead import MainBead
from qiskit_nature.problems.sampling.protein_folding.peptide.chains.side_chain import SideChain
from test.problems.sampling.protein_folding.resources.file_parser import read_expected_file

PATH = "problems/sampling/protein_folding/resources/test_bead_main"


class TestMainBead(QiskitNatureTestCase):
    """Tests Main Bead."""

    def test_main_bead_constructor(self):
        """Tests that a MainBead is created."""
        main_chain_len = 4
        num_turn_qubits = 2 * (main_chain_len - 1)
        main_bead_id = 3
        residue_type = "S"
        turn_qubits = [
            0.5 * _build_full_identity(num_turn_qubits) - 0.5 * (I ^ Z ^ I ^ I ^ I ^ I),
            0.5 * _build_full_identity(num_turn_qubits) - 0.5 * (Z ^ I ^ I ^ I ^ I ^ I),
        ]
        side_chain_len = 1
        side_chain_residue_sequences = ["S"]
        side_chain = SideChain(
            main_chain_len, main_bead_id, side_chain_len, side_chain_residue_sequences
        )
        main_bead = MainBead(main_bead_id, residue_type, turn_qubits, side_chain)

        assert main_bead.side_chain == side_chain
        indic_0, indic_1, indic_2, indic_3 = main_bead.get_indicator_functions()
        expected_path_indic_0 = self.get_resource_path(
            "test_main_bead_constructor_expected_indic_0",
            PATH,
        )
        expected_indic_0 = read_expected_file(expected_path_indic_0)

        expected_path_indic_1 = self.get_resource_path(
            "test_main_bead_constructor_expected_indic_1",
            PATH,
        )
        expected_indic_1 = read_expected_file(expected_path_indic_1)

        expected_path_indic_2 = self.get_resource_path(
            "test_main_bead_constructor_expected_indic_2",
            PATH,
        )
        expected_indic_2 = read_expected_file(expected_path_indic_2)

        expected_path_indic_3 = self.get_resource_path(
            "test_main_bead_constructor_expected_indic_3",
            PATH,
        )
        expected_indic_3 = read_expected_file(expected_path_indic_3)

        assert indic_0 == expected_indic_0
        assert indic_1 == expected_indic_1
        assert indic_2 == expected_indic_2
        assert indic_3 == expected_indic_3
