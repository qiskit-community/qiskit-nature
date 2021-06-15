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
"""Tests SideChain."""
from test import QiskitNatureTestCase
from problems.sampling.protein_folding.exceptions.invalid_side_chain_exception import (
    InvalidSideChainException,
)
from qiskit_nature.problems.sampling.protein_folding.peptide.chains.side_chain import SideChain


class TestSideChain(QiskitNatureTestCase):
    """Tests SideChain."""

    def test_side_chain_constructor(self):
        """Tests that a SideChain is created."""
        main_chain_len = 4
        main_bead_id = 3
        side_chain_len = 1
        side_chain_residue_seq = ["A"]
        side_chain = SideChain(main_chain_len, main_bead_id, side_chain_len, side_chain_residue_seq)
        assert len(side_chain) == 1
        assert side_chain[0].chain_type == "side_chain"
        assert side_chain[0].main_index == 3

    def test_side_chain_constructor_too_long(self):
        """Tests that a SideChain of length greater than 1 throws an exception."""
        main_chain_len = 4
        main_bead_id = 3
        side_chain_len = 2
        side_chain_residue_seq = ["S", "A"]
        with self.assertRaises(InvalidSideChainException):
            _ = SideChain(main_chain_len, main_bead_id, side_chain_len, side_chain_residue_seq)
