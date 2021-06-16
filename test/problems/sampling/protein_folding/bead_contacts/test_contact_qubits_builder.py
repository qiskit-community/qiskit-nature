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
"""Tests ContactQubitsBuilder."""
from test import QiskitNatureTestCase
import numpy as np
from qiskit.opflow import I, Z

from problems.sampling.protein_folding.bead_contacts import contact_map_builder
from problems.sampling.protein_folding.bead_contacts.contact_map import ContactMap
from problems.sampling.protein_folding.bead_distances.distance_map import DistanceMap
from problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from problems.sampling.protein_folding.peptide.peptide import Peptide
from test.problems.sampling.protein_folding.resources.file_parser import read_expected_file

PATH = "problems/sampling/protein_folding/resources/test_contact_qubits_builder"


class TestContactQubitsBuilder(QiskitNatureTestCase):
    """Tests ContactQubitsBuilder."""

    def test_create_pauli_for_contacts(self):
        """
        Tests that Pauli operators for contact qubits are created correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "S", "S"]
        main_chain_len = 5
        side_chains = [0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", None]
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chains, side_chain_residue_sequences
        )
        (
            lower_main_upper_main,
            lower_side_upper_main,
            lower_main_upper_side,
            lower_side_upper_side,
            r_contact,
        ) = contact_map_builder._create_contact_qubits(peptide)

        assert lower_main_upper_main == {}
        assert lower_side_upper_main == {}
        assert lower_main_upper_side == {}
        assert lower_side_upper_side == {}
        assert r_contact == 0

    def test_create_pauli_for_contacts_2(self):
        """
        Tests that Pauli operators for contact qubits are created correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "S"]
        main_chain_len = 6
        side_chains = [0, 0, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "S", None]
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chains, side_chain_residue_sequences
        )
        (
            lower_main_upper_main,
            lower_side_upper_main,
            lower_main_upper_side,
            lower_side_upper_side,
            r_contact,
        ) = contact_map_builder._create_contact_qubits(peptide)

        assert lower_main_upper_main == {
            1: {
                6: 0.5
                * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I)
                - 0.5
                * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ Z)
            }
        }
        assert lower_side_upper_main == {
            1: {
                5: 0.5
                * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I)
                - 0.5
                * (I ^ I ^ I ^ I ^ I ^ Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z)
            }
        }
        assert lower_main_upper_side == {}
        assert lower_side_upper_side == {}
        assert r_contact == 2

    # TODO
    def test_create_pauli_for_contacts_3(self):
        """
        Tests that Pauli operators for contact qubits are created correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "S", "S"]
        main_chain_len = 7
        side_chains = [0, 0, 1, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "S", "A", None]
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chains, side_chain_residue_sequences
        )
        (
            lower_main_upper_main,
            lower_side_upper_main,
            lower_main_upper_side,
            lower_side_upper_side,
            r_contact,
        ) = contact_map_builder._create_contact_qubits(peptide)

        # 0.5 * IIIIIIIIIIIIIIIIIIIIIIII
        # - 0.5 * IIIIIIIIIIIIIIIIIIZIIIIZ
        # ########
        # 0.5 * IIIIIIIIIIIIIIIIIIIIIIII
        # - 0.5 * IIIIIIIIIIIIIIIIIZIIIIZI
        # ########
        # 0.5 * IIIIIIIIIIIIIIIIIIIIIIII
        # - 0.5 * IIIIIIIZIIIIIIIIIIIIIIIZ
        # ########
        # 0.5 * IIIIIIIIIIIIIIIIIIIIIIII
        # - 0.5 * IIIIIIZIIIIIIIIIIIIIIIZI
        # ########
        # 0.5 * IIIIIIIIIIIIIIIIIIIIIIII
        # - 0.5 * IIIIIIIIIZIIIIIIIZIIIIII
        # ########
        # 0.5 * IIIIIIIIIIIIIIIIIIIIIIII
        # - 0.5 * IIIIIIZIIZIIIIIIIIIIIIII
        # ########

        # assert pauli_contacts == {1: {
        #     0: {4: {}, 5: {1: PauliOp(Pauli('IIIIIIIZIIIIIIIIIIIIIIIZ'), coeff=1.0)},
        #         6: {0: PauliOp(Pauli('IIIIIIIIIIIIIIIIIIZIIIIZ'), coeff=1.0)}, 7: {}},
        #     1: {4: {}, 5: {}, 6: {}, 7: {}}}, 2: {
        #     0: {5: {}, 6: {1: PauliOp(Pauli('IIIIIIZIIIIIIIIIIIIIIIZI'), coeff=1.0)},
        #         7: {0: PauliOp(Pauli('IIIIIIIIIIIIIIIIIZIIIIZI'), coeff=1.0)}},
        #     1: {5: {}, 6: {}, 7: {}}}, 3: {0: {6: {}, 7: {}}, 1: {
        #     6: {1: PauliOp(Pauli('IIIIIIZIIZIIIIIIIIIIIIII'), coeff=1.0)},
        #     7: {0: PauliOp(Pauli('IIIIIIIIIZIIIIIIIZIIIIII'), coeff=1.0)}}}}
        assert r_contact == 6

    def test_create_qubit_list(self):
        """
        Tests that the list of all qubits (conformation and interaction) is created correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "S", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 0, 0]
        side_chain_residue_sequences = [None, None, "A", None, None]

        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        contact_map = ContactMap(peptide)
        new_qubits = contact_map.create_peptide_qubit_list()

        expected_path = self.get_resource_path(
            "test_create_new_qubit_list_1",
            PATH,
        )

        expected_1 = read_expected_file(expected_path)
        expected_path = self.get_resource_path(
            "test_create_new_qubit_list_2",
            PATH,
        )

        expected_2 = read_expected_file(expected_path)
        expected_path = self.get_resource_path(
            "test_create_new_qubit_list_3",
            PATH,
        )

        expected_3 = read_expected_file(expected_path)
        expected_path = self.get_resource_path(
            "test_create_new_qubit_list_4",
            PATH,
        )

        expected_4 = read_expected_file(expected_path)
        expected_path = self.get_resource_path(
            "test_create_new_qubit_list_5",
            PATH,
        )

        expected_5 = read_expected_file(expected_path)

        assert len(new_qubits) == 6
        assert new_qubits[0] == 0
        assert new_qubits[1] == expected_1
        assert new_qubits[2] == expected_2
        assert new_qubits[3] == expected_3
        assert new_qubits[4] == expected_4
        assert new_qubits[5] == expected_5

    # TODO compare again with the original code after uncommenting energy terms
    def test_first_neighbor(self):
        """
        Tests that Pauli operators for 1st neighbour interactions are created correctly.
        """

        main_chain_residue_seq = ["S", "A", "A", "S", "S", "S"]
        main_chain_len = 6
        side_chain_lens = [0, 0, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "S", "S", None]

        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        lambda_1 = 2
        lower_main_bead_index = 1
        upper_main_bead_index = 4
        side_chain_lower_main_bead = 0
        side_chain_upper_main_bead = 0
        distance_map = DistanceMap(peptide)
        expr = distance_map._first_neighbor(
            peptide,
            lower_main_bead_index,
            side_chain_upper_main_bead,
            upper_main_bead_index,
            side_chain_lower_main_bead,
            lambda_1,
            pair_energies,
        )

        expected_path = self.get_resource_path(
            "test_first_neighbour",
            PATH,
        )
        expected = read_expected_file(expected_path)
        assert expr == expected

    # TODO compare again with the original code after uncommenting energy terms
    def test_first_neighbor_side(self):
        """
        Tests that Pauli operators for 1st neighbour interactions are created correctly.
        """

        main_chain_residue_seq = ["S", "A", "A", "S", "S", "S"]
        main_chain_len = 6
        side_chain_lens = [0, 0, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "S", "S", None]

        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        lambda_1 = 2
        lower_main_bead_index = 3
        upper_main_bead_index = 5
        side_chain_lower_main_bead = 1
        side_chain_upper_main_bead = 1
        distance_map = DistanceMap(peptide)
        expr = distance_map._first_neighbor(
            peptide,
            lower_main_bead_index,
            side_chain_upper_main_bead,
            upper_main_bead_index,
            side_chain_lower_main_bead,
            lambda_1,
            pair_energies,
        )

        expected_path = self.get_resource_path(
            "test_first_neighbour_side",
            PATH,
        )
        expected = read_expected_file(expected_path)
        assert expr == expected

    # TODO compare again with the original code after uncommenting energy terms
    def test_second_neighbor(self):
        """
        Tests that Pauli operators for 2nd neighbour interactions are created correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "S", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 0, 0]
        side_chain_residue_sequences = [None, None, "A", None, None]
        pair_energies = np.zeros((main_chain_len, 2, main_chain_len, 2))

        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        lambda_1 = 2
        lower_main_bead_index = 1
        upper_main_bead_index = 4
        side_chain_lower_main_bead = 0
        side_chain_upper_main_bead = 0
        distance_map = DistanceMap(peptide)
        expr = distance_map._second_neighbor(
            peptide,
            lower_main_bead_index,
            side_chain_upper_main_bead,
            upper_main_bead_index,
            side_chain_lower_main_bead,
            lambda_1,
            pair_energies,
        )
        expected_path = self.get_resource_path(
            "test_second_neighbour",
            PATH,
        )
        expected = read_expected_file(expected_path)
        assert expr == expected
