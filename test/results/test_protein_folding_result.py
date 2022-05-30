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
from ddt import ddt, data, unpack

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

    return ProteinFoldingResult(
        unused_qubits=protein_folding_problem.unused_qubits,
        peptide=protein_folding_problem.peptide,
        best_sequence=best_sequence,
    )


@ddt
class TestProteinFoldingResult(QiskitNatureTestCase):
    """Tests ProteinFoldingResult."""

    @unpack
    @data(
        (
            "APRLRFY",
            [""] * 7,
            "101100011",
            "1******0*************************************"
            + "*********************************************"
            + "*******************************************110001*1****",
            [False, False, False, False, False, False, False],
            (167, [4, 6, 7, 8, 9, 10, 11, 137, 144]),
            7,
        ),
        (
            "APRLR",
            ["", "", "F", "Y", ""],
            "0011011",
            "0011****01*1****",
            [False, False, True, True, False],
            (79, [4, 6, 7, 12, 13, 14, 15]),
            5,
        ),
        (
            "APRLR",
            ["", "F", "", "Y", ""],
            "10110110",
            "10**11**0110****",
            [False, True, False, True, False],
            (79, [4, 5, 6, 7, 10, 11, 14, 15]),
            5,
        ),
    )
    def test_result(
        self,
        main_chain,
        side_chain,
        best_sequence,
        binary_vector,
        hot_vector,
        unused_qubits_compact,
        main_chain_length,
    ):
        """Tests if ProteinFoldingResult is initialized properly and its attributes are properly set."""
        result = create_protein_folding_result(
            main_chain=main_chain, side_chains=side_chain, best_sequence=best_sequence
        )

        with self.subTest("Best Sequence"):
            self.assertEqual(result.best_sequence, best_sequence)

        with self.subTest("Binary Vector"):
            self.assertEqual(result.get_result_binary_vector(), binary_vector)

        with self.subTest("Hot Vector"):
            self.assertEqual(
                result._side_chain_hot_vector,
                hot_vector,
            )

        with self.subTest("Unused Qubits"):
            max_index, used_qubits = unused_qubits_compact
            expected_unused_qubits = [n for n in range(max_index + 1) if n not in used_qubits]
            self.assertEqual(
                result._unused_qubits,
                expected_unused_qubits,
            )

        with self.subTest("Main Chain Length"):
            self.assertEqual(result._main_chain_length, main_chain_length)
