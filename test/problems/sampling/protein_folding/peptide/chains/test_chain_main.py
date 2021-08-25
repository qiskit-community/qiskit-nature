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
"""Tests MainChain."""
from test import QiskitNatureTestCase
from qiskit_nature.problems.sampling.protein_folding.exceptions.invalid_side_chain_exception import (
    InvalidSideChainException,
)
from qiskit_nature.problems.sampling.protein_folding.peptide.chains.main_chain import MainChain


class TestMainChain(QiskitNatureTestCase):
    """Tests MainChain."""

    def test_main_chain_constructor(self):
        """Tests that a MainChain is created."""
        main_chain_residue_seq = "SASA"
        side_chain_residue_sequences = ["", "", "S", ""]
        main_chain = MainChain(main_chain_residue_seq, side_chain_residue_sequences)

        self.assertEqual(len(main_chain.beads_list), 4)
        assert main_chain[0].side_chain is None
        assert main_chain[1].side_chain is None
        assert main_chain[3].side_chain is None
        self.assertEqual(len(main_chain[2].side_chain), 1)

    def test_main_chain_illegal_side_chain_first(self):
        """Tests that an exception is thrown in case of illegal side chain."""
        main_chain_residue_seq = "SAAA"
        side_chain_residue_sequences = ["A", "", "S", ""]
        with self.assertRaises(InvalidSideChainException):
            _ = MainChain(
                main_chain_residue_seq,
                side_chain_residue_sequences,
            )

    def test_main_chain_illegal_side_chain_last(self):
        """Tests that an exception is thrown in case of illegal side chain."""
        main_chain_residue_seq = "SAAA"
        side_chain_residue_sequences = ["", "", "A", "S"]
        with self.assertRaises(InvalidSideChainException):
            _ = MainChain(
                main_chain_residue_seq,
                side_chain_residue_sequences,
            )
