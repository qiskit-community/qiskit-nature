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
from abc import ABC
from typing import List, Tuple

from qiskit.opflow import PauliOp, OperatorBase

from problems.sampling.protein_folding.exceptions.invalid_residue_exception import \
    InvalidResidueException
from qiskit_nature.problems.sampling.protein_folding.peptide.pauli_ops_builder import \
    _build_full_identity


class BaseBead(ABC):

    def __init__(self, residue_type: str, turn_qubits: List[PauliOp]):

        self._residue_type = residue_type
        if residue_type is not None and not self._is_valid_residue():
            raise InvalidResidueException(
                f"Provided residue type {residue_type} is not valid. Valid residue types are [C, "
                f"M, F, I, L, V, W, Y, A, G, T, S, N, Q, D, E, H, R, K, P].")

        self._turn_qubits = turn_qubits
        if self._residue_type is not None and self.turn_qubits is not None:
            FULL_ID = _build_full_identity(turn_qubits[0].num_qubits)
            self._indic_0 = (
                    (FULL_ID - self._turn_qubits[0]) @ (FULL_ID - self._turn_qubits[1])).reduce()
            self._indic_1 = (
                    self._turn_qubits[1] @ (self._turn_qubits[1] - 1 * self._turn_qubits[0])).reduce()
            self._indic_2 = (
                    self._turn_qubits[0] @ (self._turn_qubits[0] - 1 * self._turn_qubits[1])).reduce()
            self._indic_3 = (self._turn_qubits[0] @ self._turn_qubits[1]).reduce()

    @property
    def turn_qubits(self):
        return self._turn_qubits

    # for the turn that leads from the bead
    def get_indicator_functions(self) -> Tuple[
        OperatorBase, OperatorBase, OperatorBase, OperatorBase]:
        if self.turn_qubits is None:
            return None
        return self._indic_0, self._indic_1, self._indic_2, self._indic_3

    def _is_valid_residue(self):
        valid_residues = ['C', 'M', 'F', 'I', 'L', 'V', 'W', 'Y', 'A', 'G', 'T', 'S', 'N', 'Q', 'D',
                          'E', 'H', 'R', 'K', 'P']
        return self._residue_type in valid_residues
