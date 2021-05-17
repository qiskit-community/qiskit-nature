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
        main_chain_residue_seq = "SAA"
        main_chain_len = 3
        side_chain_lens = [0, 0, 1]
        side_chain_residue_sequences = [None, None, "A"]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)

        assert len(peptide.get_main_chain.beads_list) == 3
        assert len(peptide.get_main_chain.beads_list[2].side_chain.beads_list)
        assert peptide.get_main_chain.beads_list[0].side_chain is None
        assert peptide.get_main_chain.beads_list[1].side_chain is None
