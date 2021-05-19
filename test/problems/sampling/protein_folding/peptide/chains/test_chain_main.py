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
from problems.sampling.protein_folding.exceptions.invalid_side_chain_exception import \
    InvalidSideChainException
from qiskit_nature.problems.sampling.protein_folding.peptide.chains.main_chain import MainChain
from test import QiskitNatureTestCase


class TestMainChain(QiskitNatureTestCase):
    """Tests MainChain."""

    def test_main_chain_constructor(self):
        """Tests that a MainChain is created."""
        main_chain_len = 3
        main_chain_residue_seq = "SAS"
        side_chain_lens = [0, 0, 1]
        side_chain_residue_sequences = [None, None, "S"]
        main_chain = MainChain(main_chain_len, main_chain_residue_seq, side_chain_lens,
                               side_chain_residue_sequences)

        assert len(main_chain.beads_list) == 3
        assert main_chain[0].side_chain is None
        assert main_chain[1].side_chain is None
        assert len(main_chain[2].side_chain) == 1

    def test_main_chain_illegal_side_chain(self):
        """Tests that an exception is thrown in case of illegal side chain."""
        main_chain_len = 2
        main_chain_residue_seq = "SA"
        side_chain_lens = [1, 1]
        side_chain_residue_sequences = ["A", "S"]
        with self.assertRaises(InvalidSideChainException):
            _ = MainChain(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
