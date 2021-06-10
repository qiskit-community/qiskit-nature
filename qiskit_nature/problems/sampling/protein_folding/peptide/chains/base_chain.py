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

from qiskit.opflow import PauliOp

from qiskit_nature.problems.sampling.protein_folding.peptide.beads.base_bead import BaseBead
from qiskit_nature.problems.sampling.protein_folding.peptide.pauli_ops_builder import \
    _build_full_identity, _build_pauli_z_op


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

    @staticmethod
    def _build_turn_qubit(chain_len: int, bead_id: int) -> PauliOp:
        num_turn_qubits = 2 * (chain_len - 1)
        norm_factor = 0.5
        return norm_factor * _build_full_identity(num_turn_qubits) - norm_factor * _build_pauli_z_op(
            num_turn_qubits, [bead_id])
