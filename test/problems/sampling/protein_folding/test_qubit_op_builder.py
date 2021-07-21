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
"""Tests QubitOpBuilder."""
from test import QiskitNatureTestCase
from test.problems.sampling.protein_folding.resources.file_parser import read_expected_file
from qiskit.opflow import PauliSumOp
from qiskit_nature.problems.sampling.protein_folding.qubit_op_builder import QubitOpBuilder
from qiskit_nature.problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from qiskit_nature.problems.sampling.protein_folding.penalty_parameters import PenaltyParameters
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide

PATH = "problems/sampling/protein_folding/resources/test_qubit_op_builder"


class TestQubitOpBuilder(QiskitNatureTestCase):
    """Tests QubitOpBuilder."""

    def test_check_turns(self):
        """
        Tests that check turns operators are generate correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A"]
        main_chain_len = 6
        side_chain_lens = [0, 0, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", None]

        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        bead_2 = peptide.get_main_chain[2]
        bead_3 = peptide.get_main_chain[3]
        bead_4 = peptide.get_main_chain[4]
        side_bead_2 = bead_2.side_chain[0]
        side_bead_3 = bead_3.side_chain[0]
        side_bead_4 = bead_4.side_chain[0]

        t_23 = QubitOpBuilder._create_turn_operators(bead_2, bead_3)
        t_34 = QubitOpBuilder._create_turn_operators(bead_3, bead_4)
        t_2s3 = QubitOpBuilder._create_turn_operators(side_bead_2, bead_3)
        t_3s4s = QubitOpBuilder._create_turn_operators(side_bead_3, side_bead_4)

        expected_path_t23 = self.get_resource_path(
            "test_check_turns_expected_t23",
            PATH,
        )
        expected_t23 = read_expected_file(expected_path_t23)
        expected_path_t34 = self.get_resource_path(
            "test_check_turns_expected_t34",
            PATH,
        )
        expected_t34 = read_expected_file(expected_path_t34)
        expected_path_t2s3 = self.get_resource_path(
            "test_check_turns_expected_t2s3",
            PATH,
        )
        expected_t_2s3 = read_expected_file(expected_path_t2s3)
        expected_path_t3s4s = self.get_resource_path(
            "test_check_turns_expected_t3s4s",
            PATH,
        )
        expected_t_3s4s = read_expected_file(expected_path_t3s4s)
        self.assertEqual(t_23, expected_t23)
        self.assertEqual(t_34, expected_t34)
        self.assertEqual(t_2s3, expected_t_2s3)
        self.assertEqual(t_3s4s, expected_t_3s4s)

    def test_build_qubit_op(self):
        """Tests if a total Hamiltonian qubit operator is built correctly."""
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        lambda_contacts = 10
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A", "S", "A", "A"]
        main_chain_len = 9
        side_chain_lens = [0, 0, 1, 1, 1, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", "A", "A", "A", None]

        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        mj_interaction = MiyazawaJerniganInteraction()
        penalty_params = PenaltyParameters(lambda_chiral, lambda_back, lambda_1, lambda_contacts)
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        qubit_op = qubit_op_builder._build_qubit_op()
        expected_path = self.get_resource_path(
            "test_build_qubit_op_expected",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(qubit_op, expected)

    def test_build_qubit_op_2(self):
        """Tests if a total Hamiltonian qubit operator is built correctly."""
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        lambda_contacts = 10
        main_chain_residue_seq = ["S", "A", "A", "C", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", None]

        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        mj_interaction = MiyazawaJerniganInteraction()
        penalty_params = PenaltyParameters(lambda_chiral, lambda_back, lambda_1, lambda_contacts)
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        qubit_op = qubit_op_builder._build_qubit_op()
        expected_path = self.get_resource_path(
            "test_build_qubit_op_2_expected",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(qubit_op, expected)

    def test_create_h_back(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "S", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 0, 0, 0]
        side_chain_residue_sequences = [None, None, None, None, None]

        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        h_back = qubit_op_builder._create_h_back()
        expected_path = self.get_resource_path(
            "test_create_h_back_expected",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(h_back, expected)

    def test_create_h_back_side_chains(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly in the presence of side
        chains which should not have any influence in this case.
        """
        main_chain_residue_seq = ["S", "A", "A", "S", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", None]

        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        h_back = qubit_op_builder._create_h_back()
        expected_path = self.get_resource_path(
            "test_create_h_back_side_chains_expected",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(h_back, expected)

    def test_create_h_chiral(self):
        """
        Tests that the Hamiltonian chirality constraints is created correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A", "S", "A"]
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", "A", None]
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        h_chiral = qubit_op_builder._create_h_chiral()
        expected_path = self.get_resource_path(
            "test_create_h_chiral_expected",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(h_chiral, expected)

    def test_create_h_chiral_2(self):
        """
        Tests that the Hamiltonian chirality constraints is created correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "C", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", None]
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        h_chiral = qubit_op_builder._create_h_chiral()
        expected_path = self.get_resource_path(
            "test_create_h_chiral_2_expected",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(h_chiral, expected)

    def test_create_h_bbbb(self):
        """Tests if H_BBBB Hamiltonian qubit operator is built correctly."""
        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A", "S", "A"]
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", "A", None]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        h_bbbb = qubit_op_builder._create_h_bbbb()
        expected_path = self.get_resource_path(
            "test_create_h_bbbb_expected",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(h_bbbb, expected)

    def test_create_h_bbbb_2(self):
        """
        Tests if H_BBBB Hamiltonian qubit operator is built correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "C", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", None]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        h_bbbb = qubit_op_builder._create_h_bbbb()
        expected = 0
        self.assertEqual(h_bbbb, expected)

    def test_create_h_bbsc_and_h_scbb(self):
        """Tests if H_BBSC and H_SCBB Hamiltonians qubit operators are built correctly."""
        main_chain_residue_seq = ["A", "P", "R", "L", "A", "A", "A"]
        main_chain_len = 7
        side_chain_lens = [0, 0, 1, 0, 0, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", None]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)

        h_bbsc, h_scbb = qubit_op_builder._create_h_bbsc_and_h_scbb()

        expected_path_h_bbsc = self.get_resource_path(
            "test_create_h_bbsc_and_h_scbb_expected_h_bbsc",
            PATH,
        )
        expected_path_h_scbb = self.get_resource_path(
            "test_create_h_bbsc_and_h_scbb_expected_h_scbb",
            PATH,
        )
        expected_h_bbsc = read_expected_file(expected_path_h_bbsc)
        expected_h_scbb = read_expected_file(expected_path_h_scbb)
        self.assertEqual(h_bbsc, expected_h_bbsc)
        self.assertEqual(h_scbb, expected_h_scbb)

    def test_create_h_bbsc_and_h_scbb_2(self):
        """Tests if H_BBSC and H_SCBB Hamiltonians qubit operators are built correctly."""
        main_chain_residue_seq = ["A", "P", "R", "L", "A", "A"]
        main_chain_len = 6
        side_chain_lens = [0, 0, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", None]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)

        h_bbsc, h_scbb = qubit_op_builder._create_h_bbsc_and_h_scbb()

        expected_path_h_bbsc = self.get_resource_path(
            "test_create_h_bbsc_and_h_scbb_2_expected",
            PATH,
        )
        expected_h_bbsc = read_expected_file(expected_path_h_bbsc)
        self.assertEqual(h_bbsc, expected_h_bbsc)
        self.assertEqual(h_scbb, 0)

    def test_create_h_bbsc_and_h_scbb_3(self):
        """
        Tests if H_BBBB Hamiltonian qubit operator is built correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "C", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", None]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        h_bbsc, h_scbb = qubit_op_builder._create_h_bbsc_and_h_scbb()
        self.assertEqual(h_bbsc, 0)
        self.assertEqual(h_scbb, 0)

    def test_create_h_short(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly.
        """
        main_chain_residue_seq = ["A", "P", "R", "L", "A", "A", "A", "A"]
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 1, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", "A", "A", None]
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        h_short = qubit_op_builder._create_h_short()
        expected = PauliSumOp.from_list([("IIIIIIIIIIIIIIIIIIIIIIIIIIII", 0)])
        self.assertEqual(h_short, expected)

    def test_create_h_contacts(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly.
        """
        main_chain_residue_seq = ["A", "P", "R", "L", "A", "A", "A", "A"]
        main_chain_len = 8
        side_chain_lens = [0, 0, 1, 0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", None, None, "A", "A", None]
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        penalty_params = PenaltyParameters()
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        h_contacts = qubit_op_builder._create_h_contacts()
        expected_path_h_contacts = self.get_resource_path(
            "test_create_h_contacts_expected",
            PATH,
        )
        expected = read_expected_file(expected_path_h_contacts)
        self.assertEqual(h_contacts, expected)

    def test_create_h_contacts_2(self):
        """
        Tests that the Hamiltonian to back-overlaps is created correctly.
        """
        main_chain_residue_seq = ["S", "A", "A", "C", "S"]
        main_chain_len = 5
        side_chain_lens = [0, 0, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", None]
        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )
        mj_interaction = MiyazawaJerniganInteraction()
        pair_energies = mj_interaction.calc_energy_matrix(main_chain_len, main_chain_residue_seq)
        penalty_params = PenaltyParameters()
        qubit_op_builder = QubitOpBuilder(peptide, pair_energies, penalty_params)
        h_contacts = qubit_op_builder._create_h_contacts()
        expected_path_h_contacts = self.get_resource_path(
            "test_create_h_contacts_2_expected",
            PATH,
        )
        expected = read_expected_file(expected_path_h_contacts)
        self.assertEqual(h_contacts, expected)
