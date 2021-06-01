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
from typing import List

from qiskit.opflow import PauliOp

from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_full_identity
from qiskit_nature.problems.sampling.protein_folding.peptide.beads.base_bead import BaseBead
from qiskit_nature.problems.sampling.protein_folding.peptide.chains.side_chain import SideChain


class MainBead(BaseBead):

    def __init__(self, residue_type: str, turn_qubits: List[PauliOp], side_chain: SideChain):
        super().__init__(residue_type, turn_qubits)
        self._side_chain = side_chain
        if self._residue_type is not None and self.turn_qubits is not None:
            FULL_ID = _build_full_identity(turn_qubits[0].num_qubits)
            self._indic_0 = (FULL_ID ^
                             (FULL_ID - self._turn_qubits[0]) @ (
                                         FULL_ID - self._turn_qubits[1])).reduce()
            self._indic_1 = (FULL_ID ^
                             self._turn_qubits[1] @ (
                                     self._turn_qubits[1] - 1 * self._turn_qubits[0])).reduce()
            self._indic_2 = (FULL_ID ^
                             self._turn_qubits[0] @ (
                                     self._turn_qubits[0] - 1 * self._turn_qubits[1])).reduce()
            self._indic_3 = (FULL_ID ^ self._turn_qubits[0] @ self._turn_qubits[1]).reduce()

    @property
    def side_chain(self) -> SideChain:
        return self._side_chain
