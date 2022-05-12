# This code is part of Qiskit.
#
# (C) Copyright IBM 2021,2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests ProteinFoldingResult."""
from typing import List

from test import QiskitNatureTestCase
from qiskit.utils import algorithm_globals
from qiskit_nature.problems.sampling.protein_folding.protein_folding_problem import (
    ProteinFoldingProblem,
)
from qiskit_nature.results.protein_folding_result import ProteinFoldingResult
from qiskit_nature.problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import (
    MiyazawaJerniganInteraction,
)
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide

from qiskit_nature.problems.sampling.protein_folding.penalty_parameters import PenaltyParameters


def create_protein_folding_result(
    main_chain: str, side_chains: List[str], best_sequence: str
) -> ProteinFoldingResult:
    """
    Creates a protein_folding_problem, solves it and uses the result
    to create a protein_folding_result instance.
    Args:
        main_chain: The desired main_chain for the molecules to be optimized
        side_chains: The desired side_chains for the molecules to be optimized
        best_sequence: The best sequence found by ProteinFoldingResult pre-computed
    Returns:
        Protein Folding Result
    """
    algorithm_globals.random_seed = 23
    peptide = Peptide(main_chain, side_chains)
    mj_interaction = MiyazawaJerniganInteraction()

    penalty_back = 10
    penalty_chiral = 10
    penalty_1 = 10

    penalty_terms = PenaltyParameters(penalty_chiral, penalty_back, penalty_1)

    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    protein_folding_problem.qubit_op()

    return ProteinFoldingResult(protein_folding_problem, best_sequence)


class TestProteinFoldingResult(QiskitNatureTestCase):
    """Tests ProteinFoldingResult."""

    test_result_1 = create_protein_folding_result("APRLRFY", [""] * 7, "101100011")
    test_result_2 = create_protein_folding_result("APRLR", ["", "", "F", "Y", ""], "0011011")
    test_result_3 = create_protein_folding_result("APRLR", ["", "F", "", "Y", ""], "10110110")

    def test_best_sequence(self):
        """Tests if the best sequence obtained is the correct one and if
        it gets passed to the constructor correctly."""
        # Tests for case 1
        self.assertEqual(self.test_result_1.best_sequence, "101100011")
        # Tests for case 2
        self.assertEqual(self.test_result_2.best_sequence, "0011011")
        # Tests for case 3
        self.assertEqual(self.test_result_3.best_sequence, "10110110")

    def test_binary_vector(self):
        """Tests if the result binary vector is expanded correctly"""
        # Test for case 1
        self.assertEqual(
            self.test_result_1.get_result_binary_vector(),
            "1******0*************************************"
            + "*********************************************"
            + "*******************************************110001*1****",
        )
        # Test for case 2
        self.assertEqual(self.test_result_2.get_result_binary_vector(), "0011****01*1****")
        # Test for case 3
        self.assertEqual(self.test_result_3.get_result_binary_vector(), "10**11**0110****")

    def test_side_chain_hot_vector(self):
        """Tests if the hot vector from the side chain is correct"""
        # Test for case 1
        self.assertEqual(
            self.test_result_1._side_chain_hot_vector,
            [False, False, False, False, False, False, False],
        )
        # Test for case 2
        self.assertEqual(
            self.test_result_2._side_chain_hot_vector, [False, False, True, True, False]
        )
        # Test for case 3
        self.assertEqual(
            self.test_result_3._side_chain_hot_vector, [False, True, False, True, False]
        )

    def test_unused_qubits(self):
        """Tests the list of unused qubits"""
        # Test for case 1
        max_index_1, used_qubits_1 = 167, [4, 6, 7, 8, 9, 10, 11, 137, 144]
        expected_unused_qubits_1 = [n for n in range(max_index_1 + 1) if n not in used_qubits_1]
        self.assertEqual(
            self.test_result_1._unused_qubits,
            expected_unused_qubits_1,
        )

        # Test for case 2
        max_index_2, used_qubits_2 = 79, [4, 6, 7, 12, 13, 14, 15]
        expected_unused_qubits_2 = [n for n in range(max_index_2 + 1) if n not in used_qubits_2]
        self.assertEqual(
            self.test_result_2._unused_qubits,
            expected_unused_qubits_2,
        )

        # Test for case 3
        max_index_3, used_qubits_3 = 79, [4, 5, 6, 7, 10, 11, 14, 15]
        expected_unused_qubits_3 = [n for n in range(max_index_3 + 1) if n not in used_qubits_3]

        self.assertEqual(
            self.test_result_3._unused_qubits,
            expected_unused_qubits_3,
        )

    def test_main_chain_length(self):
        """Tests the main chain length"""
        # Test for case 1
        self.assertEqual(self.test_result_1._main_chain_length, 7)
        # Test for case 2
        self.assertEqual(self.test_result_2._main_chain_length, 5)
        # Test for case 3
        self.assertEqual(self.test_result_2._main_chain_length, 5)
