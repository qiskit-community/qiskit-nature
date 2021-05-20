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
from qiskit.opflow import PauliOp, I, Z

from problems.sampling.protein_folding.builders.qubit_op_builder import _create_h_back, \
    _create_H_chiral
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from test import QiskitNatureTestCase


class TestContactQubitsBuilder(QiskitNatureTestCase):
    """Tests ContactQubitsBuilder."""

    def test_check_turns(self) -> PauliOp:
        """

        """
        pass

    def test_create_h_back(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly.
        """
        lambda_back = 2
        main_chain_residue_seq = "SAASS"
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 0, 0]
        side_chain_residue_sequences = [None, None, "A", None, None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        h_back = _create_h_back(peptide, lambda_back)
        assert h_back == 1.5 * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 0.5 * (
                I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I) \
               + 0.5 * (I ^ I ^ I ^ Z ^ I ^ Z ^ I ^ I) + 1.0 * (I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I) + \
               0.5 * (I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I) + 1.0 * (I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I) + \
               0.5 * (I ^ I ^ I ^ I ^ I ^ Z ^ I ^ Z) + 0.5 * (I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z)

    def test_create_H_chiral(self):
        """
        Tests that the Hamiltonian chirality constraints is created correctly.
        """
        lambda_chiral = 3
        main_chain_residue_seq = "SAASS"
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 0, 0]
        side_chain_residue_sequences = [None, None, "A", None, None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        H_chiral = _create_H_chiral(peptide, lambda_chiral)
        assert H_chiral == 1.875 * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.375 * (
                Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I) + 0.375 * (
                       Z ^ I ^ I ^ I ^ I ^ Z ^ I ^ I) - 0.375 * (
                       I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I) - 0.375 * (
                       Z ^ I ^ I ^ I ^ I ^ I ^ Z ^ I) - 1.125 * (
                       I ^ I ^ I ^ I ^ Z ^ I ^ Z ^ I) - 0.375 * (
                       I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I) + 0.375 * (Z ^ I ^ I ^ I ^ Z ^ Z ^ Z ^ I)
