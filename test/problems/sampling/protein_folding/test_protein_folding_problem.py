# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests ProteinFoldingProblem."""
from test import QiskitNatureDeprecatedTestCase
from test.problems.sampling.protein_folding.resources.file_parser import read_expected_file
from qiskit_nature.problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from qiskit_nature.problems.sampling.protein_folding.protein_folding_problem import (
    ProteinFoldingProblem,
)
from qiskit_nature.problems.sampling.protein_folding.penalty_parameters import PenaltyParameters

PATH = "problems/sampling/protein_folding/resources/test_protein_folding_problem"


class TestProteinFoldingProblem(QiskitNatureDeprecatedTestCase):
    """Tests ProteinFoldingProblem."""

    def test_protein_folding_problem(self):
        """Tests if a protein folding problem is created and returns a correct qubit operator."""
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        penalty_terms = PenaltyParameters(lambda_chiral, lambda_back, lambda_1)

        main_chain_residue_seq = "SAASSASAAG"
        side_chain_residue_sequences = ["", "", "A", "A", "A", "A", "A", "A", "S", ""]

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)

        mj_interaction = MiyazawaJerniganInteraction()

        protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
        qubit_op = protein_folding_problem._qubit_op_full()

        expected_path = self.get_resource_path(
            "test_protein_folding_problem",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(qubit_op, expected)

    def test_protein_folding_problem_2(self):
        """Tests if a protein folding problem is created and returns a correct qubit operator."""
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        penalty_terms = PenaltyParameters(lambda_chiral, lambda_back, lambda_1)

        main_chain_residue_seq = "SAAS"
        side_chain_residue_sequences = ["", "", "A", ""]

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)

        mj_interaction = MiyazawaJerniganInteraction()

        protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
        qubit_op = protein_folding_problem._qubit_op_full()

        expected_path = self.get_resource_path(
            "test_protein_folding_problem_2",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(qubit_op, expected)

    def test_protein_folding_problem_2_second_bead_side_chain(self):
        """Tests if a protein folding problem is created and returns a correct qubit operator if
        a second main bead has a side chain."""
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        penalty_terms = PenaltyParameters(lambda_chiral, lambda_back, lambda_1)

        main_chain_residue_seq = "SAAS"
        side_chain_residue_sequences = ["", "A", "A", ""]

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)

        mj_interaction = MiyazawaJerniganInteraction()

        protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
        qubit_op = protein_folding_problem._qubit_op_full()

        expected_path = self.get_resource_path(
            "test_protein_folding_problem_2_second_bead_side_chain",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(qubit_op, expected)
