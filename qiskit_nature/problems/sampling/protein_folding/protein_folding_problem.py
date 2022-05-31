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
"""Defines a protein folding problem that can be passed to algorithms."""
from __future__ import annotations

from typing import Union, List

from qiskit.opflow import PauliSumOp, PauliOp
from qiskit.algorithms import MinimumEigensolverResult

from qiskit_nature.results.protein_folding_result import ProteinFoldingResult
from .peptide.peptide import Peptide
from .interactions.interaction import Interaction
from .penalty_parameters import PenaltyParameters
from .qubit_op_builder import QubitOpBuilder
from .qubit_utils import qubit_number_reducer
from ..sampling_problem import SamplingProblem


class ProteinFoldingProblem(SamplingProblem):
    """Defines a protein folding problem that can be passed to algorithms. Example initialization:

    .. code-block:: python

        penalty_terms = PenaltyParameters(15, 15, 15)
        main_chain_residue_seq = "SAASSASAAG"
        side_chain_residue_sequences = ["", "", "A", "A", "A", "A", "A", "A", "S", ""]
        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)
        mj_interaction = MiyazawaJerniganInteraction()
        protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
        qubit_op = protein_folding_problem.qubit_op()

    """

    def __init__(
        self, peptide: Peptide, interaction: Interaction, penalty_parameters: PenaltyParameters
    ):
        """
        Args:
            peptide: A peptide object that defines the protein subject to the folding problem.
            interaction: A type of interaction between the beads of the peptide.
            penalty_parameters: Parameters that define the strength of constraints enforcing in
                                the problem.
        """
        self._peptide = peptide
        self._interaction = interaction
        self._penalty_parameters = penalty_parameters
        self._pair_energies = interaction.calculate_energy_matrix(
            peptide.get_main_chain.main_chain_residue_sequence
        )
        self._qubit_op_builder = QubitOpBuilder(
            self._peptide, self._pair_energies, self._penalty_parameters
        )
        self._unused_qubits: List[int] = []

    def qubit_op(self) -> Union[PauliSumOp, PauliOp]:
        """
        Builds a qubit operator for the Hamiltonian encoding a protein folding problem. The
        number of qubits needed for optimization is optimized (compressed), if possible.
        To obtain the full qubit operator for a Hamiltonian, use the method `qubit_op_full`.

        Returns:
            A qubit operator for the Hamiltonian encoding a protein folding problem on an
            optimized number of qubits.
        """
        qubit_operator, unused_qubits = qubit_number_reducer._remove_unused_qubits(
            self._qubit_op_full()
        )
        self._unused_qubits = unused_qubits
        return qubit_operator

    def _qubit_op_full(self) -> Union[PauliOp, PauliSumOp]:
        """
        Builds a full qubit operator for the Hamiltonian encoding a protein folding problem. Full
        means that the number of qubits needed for optimization is not optimized and may be
        larger that necessary. To ensure the optimal number of qubits, use the method `qubit_op`.

        Returns:
            A qubit operator for the Hamiltonian encoding a protein folding problem.
        """
        qubit_operator = self._qubit_op_builder._build_qubit_op()
        return qubit_operator

    def interpret(self, raw_result: MinimumEigensolverResult) -> ProteinFoldingResult:
        """
        Returns a ProteinFoldingResult object that will allow us to interpret the result obtained.
        For now we are only interested in the sequence with the biggest amplitude from the eigenstate.
        """
        best_turns_sequence = max(raw_result.eigenstate, key=raw_result.eigenstate.get)
        return ProteinFoldingResult(
            unused_qubits=self.unused_qubits, peptide=self.peptide, turns_sequence=best_turns_sequence
        )

    @property
    def unused_qubits(self) -> List[int]:
        """Returns the list of indices for qubits in the original problem formulation that were
        removed during compression."""
        return self._unused_qubits

    @property
    def peptide(self) -> Peptide:
        """Returns the peptide defining the protein subject to the folding problem."""
        return self._peptide
