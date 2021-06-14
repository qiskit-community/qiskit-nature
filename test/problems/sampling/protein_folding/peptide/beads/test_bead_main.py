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
from qiskit.opflow import I, Z

from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_full_identity
from qiskit_nature.problems.sampling.protein_folding.peptide.beads.main_bead import MainBead
from qiskit_nature.problems.sampling.protein_folding.peptide.chains.side_chain import SideChain
from test import QiskitNatureTestCase


class TestMainBead(QiskitNatureTestCase):
    """Tests Main Bead."""

    def test_main_bead_constructor(self):
        """Tests that a MainBead is created."""
        main_chain_len = 4
        num_turn_qubits = 2 * (main_chain_len - 1)
        main_bead_id = 3
        residue_type = "S"
        turn_qubits = [0.5 * _build_full_identity(num_turn_qubits) - 0.5 * (I ^ Z ^ I ^ I ^ I ^ I),
                       0.5 * _build_full_identity(num_turn_qubits) - 0.5 * (Z ^ I ^ I ^ I ^ I ^ I)]
        side_chain_len = 1
        side_chain_residue_sequences = ["S"]
        side_chain = SideChain(main_chain_len, main_bead_id, side_chain_len,
                               side_chain_residue_sequences)
        main_bead = MainBead(main_bead_id, residue_type, turn_qubits, side_chain)

        assert main_bead.side_chain == side_chain
        indic_0, indic_1, indic_2, indic_3 = main_bead.get_indicator_functions()
        assert indic_0 == 0.25 * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 0.25 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I)
        assert indic_1 == 0.25 * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.25 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 0.25 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I)
        assert indic_2 == 0.25 * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 0.25 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I) - 0.25 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) - 0.25 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I)
        assert indic_3 == 0.25 * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.25 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I) - 0.25 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I) + 0.25 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I)
