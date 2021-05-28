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
import numpy as np
from qiskit.opflow import PauliOp, I, Z, PauliSumOp

from problems.sampling.protein_folding.builders import contact_qubits_builder
from problems.sampling.protein_folding.builders.qubit_op_builder import _create_h_back, \
    _create_h_chiral, _create_h_bbbb, _create_h_bbsc_and_h_scbb, _create_h_scsc
from problems.sampling.protein_folding.distance_calculator import _calc_distances_main_chain, \
    _add_distances_side_chain, _calc_total_distances
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
        lambda_back = 10
        main_chain_residue_seq = "SAASS"
        main_chain_len = 5
        side_chain_lens = [0, 0, 0, 0, 0]
        side_chain_residue_sequences = [None, None, None, None, None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        h_back = _create_h_back(peptide, lambda_back)

        assert h_back == 2.5 * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 2.5 * (
                Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 2.5 * (I ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I) - 2.5 * (
                       Z ^ Z ^ I ^ Z ^ I ^ I ^ I ^ I)

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
        H_chiral = _create_h_chiral(peptide, lambda_chiral)
        # TODO improve PauliSumOp
        expected = 3 * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 1 * (Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 1 * (
                Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I)
        assert H_chiral == expected.reduce()

    def test_create_H_BBBB(self):
        """
        Creates Hamiltonian term corresponding to 1st neighbor interaction between
        main/backbone (BB) beads

        Args:
            main_chain_len: Number of total beads in peptide
            lambda_1: Constraint to penalize local overlap between
                     beads within a nearest neighbor contact
            pair_energies: Numpy array of pair energies for amino acids
            x_dist: Numpy array that tracks all distances between backbone and side chain
                    beads for all axes: 0,1,2,3
            pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
            contacts: Dictionary of contact qubits in symbolic notation

        Returns:
            H_BBBB: Hamiltonian term in symbolic notation
        """
        lambda_1 = 1
        main_chain_residue_seq = "SAASS"
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "S", None]
        pair_energies = np.zeros((main_chain_len, 2, main_chain_len, 2))
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        side_chain = peptide.get_side_chain_hot_vector()
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(peptide, delta_n0,
                                                                           delta_n1, delta_n2,
                                                                           delta_n3)
        x_dist = _calc_total_distances(peptide, delta_n0, delta_n1,
                                       delta_n2, delta_n3)
        contacts, r_contact = contact_qubits_builder._create_pauli_for_contacts(peptide)
        h_bbbb = _create_h_bbbb(main_chain_len, lambda_1, pair_energies,
                                x_dist, contacts)
        expected = 0 * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + 1 * (Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 1 * (
                Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I)
        assert h_bbbb == expected.reduce()

    def test_create_H_BBSC_and_H_SCBB(self):
        """
        Creates Hamiltonian term corresponding to 1st neighbor interaction between
        main/backbone (BB) and side chain (SC) beads. In the absence
        of side chains, this function returns a value of 0.

        Args:
            main_chain_len: Number of total beads in peptide
            side: List of side chains in peptide
            lambda_1: Constraint to penalize local overlap between
                     beads within a nearest neighbor contact
            pair_energies: Numpy array of pair energies for amino acids
            x_dist: Numpy array that tracks all distances between backbone and side chain
                    beads for all axes: 0,1,2,3
            pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
            contacts: Dictionary of contact qubits in symbolic notation

        Returns:
            H_BBSC, H_SCBB: Tuple of Hamiltonian terms consisting of backbone and side chain
            interactions
        """
        lambda_1 = 2
        main_chain_residue_seq = "SAASSASA"
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", "A", None]
        pair_energies = np.zeros((main_chain_len, 2, main_chain_len, 2))
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        side_chain = peptide.get_side_chain_hot_vector()
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(peptide, delta_n0,
                                                                           delta_n1, delta_n2,
                                                                           delta_n3)
        x_dist = _calc_total_distances(peptide, delta_n0, delta_n1,
                                       delta_n2, delta_n3)
        contacts, r_contact = contact_qubits_builder._create_pauli_for_contacts(peptide)
        H_BBSC, H_SCBB = _create_h_bbsc_and_h_scbb(main_chain_len, side_chain, lambda_1,
                                                   pair_energies, x_dist,
                                                   contacts)
        print(H_BBSC)
        print(H_SCBB)

    def test_create_H_SCSC(self):
        """
        Creates Hamiltonian term corresponding to 1st neighbor interaction between
        side chain (SC) beads. In the absence of side chains, this function
        returns a value of 0.

        Args:
            main_chain_len: Number of total beads in peptides
            lambda_1: Constraint to penalize local overlap between
                     beads within a nearest neighbor contact
            pair_energies: Numpy array of pair energies for amino acids
            x_dist: Numpy array that tracks all distances between backbone and side chain
                    beads for all axes: 0,1,2,3
            pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
            contacts: Dictionary of contact qubits in symbolic notation

        Returns:
            H_SCSC: Hamiltonian term consisting of side chain pairwise interactions
        """
        lambda_1 = 3
        main_chain_residue_seq = "SAASS"
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 0, 0]
        side_chain_residue_sequences = [None, None, "A", None, None]
        pair_energies = np.zeros((main_chain_len, 2, main_chain_len, 2))
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        side_chain = peptide.get_side_chain_hot_vector()
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(peptide, delta_n0,
                                                                           delta_n1, delta_n2,
                                                                           delta_n3)
        x_dist = _calc_total_distances(peptide, delta_n0, delta_n1,
                                       delta_n2, delta_n3)
        contacts, r_contact = contact_qubits_builder._create_pauli_for_contacts(peptide)
        H_SCSC = _create_h_scsc(main_chain_len, side_chain, lambda_1,
                                pair_energies, x_dist, contacts)
        print(H_SCSC)
