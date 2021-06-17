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
"""Tests ContactMapBuilder."""
from test import QiskitNatureTestCase
from test.problems.sampling.protein_folding.resources.file_parser import read_expected_file
from qiskit.opflow import I, Z

from problems.sampling.protein_folding.bead_contacts import contact_map_builder
from problems.sampling.protein_folding.bead_contacts.contact_map import ContactMap
from problems.sampling.protein_folding.peptide.peptide import Peptide

PATH = "problems/sampling/protein_folding/resources/test_contact_map_builder"


class TestContactMapBuilder(QiskitNatureTestCase):
    """Tests ContactMapBuilder."""

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
