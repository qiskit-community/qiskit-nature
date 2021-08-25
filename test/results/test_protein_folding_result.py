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
"""Tests ProteinFoldingResult."""
from test import QiskitNatureTestCase
from qiskit_nature.results.protein_folding_result import ProteinFoldingResult
from qiskit_nature.problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from qiskit_nature.problems.sampling.protein_folding.protein_folding_problem import (
    ProteinFoldingProblem,
)
from qiskit_nature.problems.sampling.protein_folding.penalty_parameters import PenaltyParameters


class TestProteinFoldingProblem(QiskitNatureTestCase):
    """Tests ProteinFoldingResult."""

    def test_get_result_binary_vector(self):
        """Tests if a protein folding result returns a correct expanded best sequence if not
        qubits compressed."""
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        penalty_terms = PenaltyParameters(lambda_chiral, lambda_back, lambda_1)

        main_chain_residue_seq = "SAASSASAA"
        side_chain_residue_sequences = ["", "", "A", "A", "A", "A", "A", "A", ""]

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)

        mj_interaction = MiyazawaJerniganInteraction()

        protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
        best_sequence = "101110010"

        protein_folding_result = ProteinFoldingResult(protein_folding_problem, best_sequence)

        result = protein_folding_result.get_result_binary_vector()

        self.assertEqual(result, best_sequence)

    def test_get_result_binary_vector_compressed(self):
        """Tests if a protein folding result returns a correct expanded best sequence."""
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        penalty_terms = PenaltyParameters(lambda_chiral, lambda_back, lambda_1)

        main_chain_residue_seq = "SAASSASAA"
        side_chain_residue_sequences = ["", "", "A", "A", "A", "A", "A", "A", ""]

        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)

        mj_interaction = MiyazawaJerniganInteraction()

        protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
        best_sequence = "101110010"
        protein_folding_problem._unused_qubits = [0, 1, 2, 5, 7]

        protein_folding_result = ProteinFoldingResult(protein_folding_problem, best_sequence)

        result = protein_folding_result.get_result_binary_vector()
        expected_sequence = "101110*0*10***"

        self.assertEqual(result, expected_sequence)
