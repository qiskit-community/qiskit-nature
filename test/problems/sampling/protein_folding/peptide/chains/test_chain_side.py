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
from qiskit_nature.problems.sampling.protein_folding.peptide.chains.side_chain import SideChain
from test import QiskitNatureTestCase


class TestSideChain(QiskitNatureTestCase):
    """Tests SideChain."""

    def test_side_chain_constructor(self):
        """Tests that a SideChain is created."""
        side_chain_len = 1
        side_chain_residue_seq = "A"
        side_chain = SideChain(side_chain_len, side_chain_residue_seq)
        print(side_chain.beads_list[0])

    def test_side_chain_constructor_too_long(self):
        """Tests that a SideChain of length greater than 1 throws an exception."""
        side_chain_len = 2
        side_chain_residue_seq = "SA"
        with self.assertRaises(Exception):
            _ = SideChain(side_chain_len, side_chain_residue_seq)
