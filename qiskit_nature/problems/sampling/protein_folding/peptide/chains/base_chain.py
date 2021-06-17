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
"""An abstract class defining a chain of a peptide."""
from abc import ABC
from typing import List

from qiskit.opflow import PauliOp

from ...peptide.beads.base_bead import BaseBead
from ...peptide.pauli_ops_builder import (
    _build_full_identity,
    _build_pauli_z_op,
)


class BaseChain(ABC):
    """An abstract class defining a chain of a peptide."""

    def __init__(self, beads_list: List[BaseBead]):
        self._beads_list = beads_list

    def __getitem__(self, item):
        return self._beads_list[item]

    def __len__(self):
        return len(self._beads_list)

    @property
    def beads_list(self) -> List[BaseBead]:
        """Returns the list of all beads in the chain."""
        return self._beads_list

    def get_residue_sequence(self) -> List[str]:
        """
        Returns the list of all residue sequences in the chain.
        """
        residue_sequence = []
        for bead in self._beads_list:
            residue_sequence.append(bead.residue_type)
        return residue_sequence

    @staticmethod
    def _build_turn_qubit(chain_len: int, bead_id: int) -> PauliOp:
        """

        Args:
            chain_len: length of the chain.
            bead_id: index of a bead in the chain.

        Returns:
            turn_qubit: A Pauli operator that encodes the turn following from a given bead index.
        """
        num_turn_qubits = 2 * (chain_len - 1)
        norm_factor = 0.5
        turn_qubit = norm_factor * _build_full_identity(
            num_turn_qubits
        ) - norm_factor * _build_pauli_z_op(num_turn_qubits, [bead_id])
        return turn_qubit
