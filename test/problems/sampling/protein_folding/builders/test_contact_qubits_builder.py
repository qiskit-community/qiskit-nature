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
from qiskit.opflow import I, Z, PauliSumOp, PauliOp
from qiskit.quantum_info import SparsePauliOp, Pauli

from problems.sampling.protein_folding.builders import contact_qubits_builder
from problems.sampling.protein_folding.builders.contact_qubits_builder import \
    _create_new_qubit_list, _first_neighbor, _second_neighbor
from problems.sampling.protein_folding.distance_calculator import _calc_total_distances, \
    _calc_distances_main_chain, _add_distances_side_chain
from problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import \
    MiyazawaJerniganInteraction
from problems.sampling.protein_folding.peptide.peptide import Peptide
from test import QiskitNatureTestCase


class TestContactQubitsBuilder(QiskitNatureTestCase):
    """Tests Peptide."""

    # TODO validate with symbolic
    def test_create_pauli_for_contacts(self):
        """
        Tests that Pauli operators for contact qubits are created correctly.
        """
        main_chain_residue_seq = "SAASS"
        main_chain_len = 5
        side_chains = [0, 0, 1, 1, 1]
        side_chain_residue_sequences = [None, None, "A", "A", "A"]
        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chains,
                          side_chain_residue_sequences)
        pauli_contacts, r_contact = contact_qubits_builder._create_pauli_for_contacts(peptide)
        assert pauli_contacts == {
            1: {0: {4: {}, 5: {1: PauliOp(Pauli('IIIZIIIIIIIIIIIZ'), coeff=1.0)}},
                1: {4: {}, 5: {}}}}
        assert r_contact == 1

    # TODO validate with symbolic
    def test_create_new_qubit_list(self):
        """
        Tests that the list of all qubits (conformation and interaction) is created correctly.
        """
        main_chain_residue_seq = "SAASS"
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 0, 0]
        side_chain_residue_sequences = [None, None, "A", None, None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        pauli_contacts, r_contact = contact_qubits_builder._create_pauli_for_contacts(peptide)
        new_qubits = _create_new_qubit_list(peptide, pauli_contacts)
        assert new_qubits[0] == 0
        assert new_qubits[1] == 0.5 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 0.5 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I)

    # TODO validate with symbolic
    def test_first_neighbor(self):
        """
        Tests that Pauli operators for 1st neighbour interactions are created correctly.
        """

        main_chain_residue_seq = "SAASSS"
        main_chain_len = 6
        side_chain_lens = [0, 0, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "S", "S", None]

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        mj = MiyazawaJerniganInteraction()
        pair_energies = mj.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        lambda_1 = 2
        lower_main_bead_index = 1
        upper_main_bead_index = 4
        side_chain_lower_main_bead = 0
        side_chain_upper_main_bead = 0
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(peptide, delta_n0,
                                                                           delta_n1, delta_n2,
                                                                           delta_n3)
        x_dist = _calc_total_distances(peptide, delta_n0, delta_n1,
                                       delta_n2, delta_n3)
        expr = _first_neighbor(lower_main_bead_index, side_chain_upper_main_bead,
                               upper_main_bead_index, side_chain_lower_main_bead, lambda_1,
                               pair_energies, x_dist)

        assert expr == 168.0 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^
                    I) + 56.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I
                           ^ I ^ I)

    # TODO validate with symbolic
    def test_second_neighbor(self):
        """
        Tests that Pauli operators for 2nd neighbour interactions are created correctly.
        """
        main_chain_residue_seq = "SAASS"
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 0, 0]
        side_chain_residue_sequences = [None, None, "A", None, None]
        pair_energies = np.zeros((main_chain_len, 2, main_chain_len, 2))

        peptide = Peptide(main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences)
        lambda_1 = 2
        lower_main_bead_index = 1
        upper_main_bead_index = 4
        side_chain_lower_main_bead = 0
        side_chain_upper_main_bead = 0
        delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
        delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(peptide, delta_n0,
                                                                           delta_n1, delta_n2,
                                                                           delta_n3)
        x_dist = _calc_total_distances(peptide, delta_n0, delta_n1,
                                       delta_n2, delta_n3)
        expr = _second_neighbor(lower_main_bead_index, side_chain_upper_main_bead,
                                upper_main_bead_index, side_chain_lower_main_bead, lambda_1,
                                pair_energies, x_dist)
        assert expr == -4.0 * (
                    I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) - 2.0 * (
                           I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I)
