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
from typing import List

from qiskit.opflow import I, Z, PauliOp

from qiskit_nature.problems.sampling.protein_folding.peptide.beads.base_bead import BaseBead
from qiskit_nature.problems.sampling.protein_folding.peptide.pauli_ops_builder import \
    _build_full_identity


class BaseChain(ABC):

    def __init__(self, beads_list: List[BaseBead]):
        self._beads_list = beads_list

    def __getitem__(self, item):
        return self._beads_list[item]

    def __len__(self):
        return len(self._beads_list)

    @property
    def beads_list(self) -> List[BaseBead]:
        return self._beads_list

    def get_residue_sequence(self):
        residue_sequence = []
        for bead in self._beads_list:
            residue_sequence.append(bead.residue_type)
        return residue_sequence

    def _build_turn_qubit(self, chain_len, bead_id) -> PauliOp:
        num_turn_qubits = 2 * (chain_len - 1)
        if bead_id != 0:
            temp = I
        else:
            temp = Z
        for i in range(1, num_turn_qubits):
            if i == bead_id:
                temp = Z ^ temp
            else:
                temp = I ^ temp
        return 0.5 * _build_full_identity(num_turn_qubits) - 0.5 * temp
