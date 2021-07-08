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
"""The protein folding result."""
from typing import Union
from qiskit.opflow import PauliOp
from qiskit_nature.results import EigenstateResult
from qiskit_nature.problems.sampling import ProteinFoldingProblem


class ProteinFoldingResult(EigenstateResult):
    """The protein folding result."""

    def __init__(
        self,
        protein_folding_problem: ProteinFoldingProblem,
        best_sequence: Union[str, PauliOp],
    ) -> None:
        super().__init__()
        self._protein_folding_problem = protein_folding_problem
        self._best_sequence: str = best_sequence

    @property
    def protein_folding_problem(self):
        """Returns the protein folding problem."""
        return self._protein_folding_problem

    @property
    def best_sequence(self):
        """Returns the best sequence."""
        return self._best_sequence

    def get_result_binary_vector(self) -> str:
        """The ProteinFoldingProblem uses a compressed optimization problem that do not match the
        number of qubits in the original objective function. This method calculates the original
        version of the solution vector. Bits that can take any value without changing the
        solution are denoted by '*'."""
        unused_qubits = self._protein_folding_problem.unused_qubits
        result = []
        offset = 0
        size = len(self._best_sequence)
        for i in range(size):
            index = size - 1 - i
            while i + offset in unused_qubits:
                result.append("*")
                offset += 1
            result.append(self._best_sequence[index])

        return "".join(result[::-1])
