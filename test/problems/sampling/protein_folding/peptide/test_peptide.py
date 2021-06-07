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
from problems.sampling.protein_folding.peptide.peptide import Peptide

from test import QiskitNatureTestCase


class TestPeptide(QiskitNatureTestCase):
    """Tests Peptide."""

    def test_peptide_constructor(self):
        """Tests that a Peptide is created."""
        main_chain_residue_seq = "SAAR"
        main_chain_len = 4
        side_chain_lens = [0, 0, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)

        side_chain_hot_vector = peptide.get_side_chain_hot_vector()

        assert len(peptide.get_main_chain.beads_list) == 4
        assert len(peptide.get_main_chain[2].side_chain.beads_list)
        assert peptide.get_main_chain[0].side_chain is None
        assert peptide.get_main_chain[1].side_chain is None
        assert peptide.get_main_chain[3].side_chain is None
        assert side_chain_hot_vector == [0, 0, 1, 0]

    def test_peptide_hot_vector_longer_chain(self):
        """Tests that a Peptide is created."""
        main_chain_residue_seq = "SAAAAAAAA"
        main_chain_len = 9
        side_chain_lens = [0, 0, 1, 0, 0, 1, 0, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", None, "A", None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)

        side_chain_hot_vector = peptide.get_side_chain_hot_vector()

        assert len(peptide.get_main_chain.beads_list) == 9
        assert side_chain_hot_vector == [0, 0, 1, 0, 0, 1, 0, 1, 0]
