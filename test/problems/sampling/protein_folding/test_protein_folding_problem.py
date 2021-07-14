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
"""Tests ProteinFoldingProblem."""
from test import QiskitNatureTestCase
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


class TestProteinFoldingProblem(QiskitNatureTestCase):
    """Tests ProteinFoldingProblem."""

    def test_protein_folding_problem(self):
        """Tests if a protein folding problem is created and returns a correct qubit operator."""
        lambda_back = 10
        lambda_chiral = 10
        lambda_1 = 10
        lambda_contacts = 10
        penalty_terms = PenaltyParameters(lambda_chiral, lambda_back, lambda_1, lambda_contacts)

        main_chain_residue_seq = ["S", "A", "A", "S", "S", "A", "S", "A", "A"]
        main_chain_len = 9
        side_chain_lens = [0, 0, 1, 1, 1, 1, 1, 1, 0]
        side_chain_residue_sequences = [None, None, "A", "A", "A", "A", "A", "A", None]

        peptide = Peptide(
            main_chain_len, main_chain_residue_seq, side_chain_lens, side_chain_residue_sequences
        )

        mj_interaction = MiyazawaJerniganInteraction()

        protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
        qubit_op = protein_folding_problem._qubit_op_full()

        expected_path = self.get_resource_path(
            "test_protein_folding_problem",
            PATH,
        )
        expected = read_expected_file(expected_path)
        self.assertEqual(qubit_op, expected)
