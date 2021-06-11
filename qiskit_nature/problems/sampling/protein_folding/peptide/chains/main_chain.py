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

from problems.sampling.protein_folding.exceptions.invalid_side_chain_exception import \
    InvalidSideChainException
from problems.sampling.protein_folding.exceptions.invalid_size_exception import InvalidSizeException
from qiskit_nature.problems.sampling.protein_folding.peptide.beads.main_bead import MainBead
from qiskit_nature.problems.sampling.protein_folding.peptide.chains.base_chain import BaseChain
from qiskit_nature.problems.sampling.protein_folding.peptide.chains.side_chain import SideChain


class MainChain(BaseChain):

    def __init__(self, main_chain_len: int, main_chain_residue_seq: List[str],
                 side_chain_lens: List[int],
                 side_chain_residue_sequences: List[str]):
        self._main_chain_residue_sequence = main_chain_residue_seq
        beads_list = self._build_main_chain(main_chain_len, main_chain_residue_seq, side_chain_lens,
                                            side_chain_residue_sequences)
        super().__init__(beads_list)

    @property
    def main_chain_residue_sequence(self):
        return self._main_chain_residue_sequence

    def _build_main_chain(self, main_chain_len: int, main_chain_residue_seq: List[str],
                          side_chain_lens: List[int],
                          side_chain_residue_sequences: List[str]) -> List[MainBead]:
        main_chain = []
        self._validate_chain_lengths(main_chain_len, side_chain_lens)
        self._validate_side_chain_index_by_lengths(side_chain_lens)
        self._validate_side_chain_index_by_residues(side_chain_residue_sequences)

        for main_bead_id in range(main_chain_len - 1):
            bead_turn_qubit_1 = self._build_turn_qubit(main_chain_len, 2 * main_bead_id)
            bead_turn_qubit_2 = self._build_turn_qubit(main_chain_len, 2 * main_bead_id + 1)
            side_chain = self._create_side_chain(main_bead_id, main_chain_len, side_chain_lens,
                                                 side_chain_residue_sequences)
            main_bead = MainBead(main_chain_residue_seq[main_bead_id],
                                 [bead_turn_qubit_1, bead_turn_qubit_2],
                                 side_chain)
            main_chain.append(main_bead)
        main_bead = MainBead(None, None, None)
        main_chain.append(main_bead)
        return main_chain

    def _validate_chain_lengths(self, main_chain_len: int, side_chain_lens):
        if side_chain_lens is not None and main_chain_len != len(side_chain_lens):
            raise InvalidSizeException("side_chain_lens size not equal main_chain_len")

    def _validate_side_chain_index_by_lengths(self, side_chain_lens: List[int]):
        if side_chain_lens is not None and (
                side_chain_lens[0] != 0 or side_chain_lens[1] != 0 or side_chain_lens[-1] != 0):
            raise InvalidSideChainException(
                "First, second and last main beads are not allowed to have a side chain. Non-zero "
                "length provided for an invalid side chain.")

    def _validate_side_chain_index_by_residues(self, side_chain_residue_sequences: List[str]):
        if side_chain_residue_sequences is not None and (
                side_chain_residue_sequences[0] is not None or side_chain_residue_sequences[
            1] is not None or side_chain_residue_sequences[-1] is not None):
            raise InvalidSideChainException(
                "First, second and last main beads are not allowed to have a side chain. Non-None "
                "residue provided for an invalid side chain")

    def _create_side_chain(self, main_bead_id: int, main_chain_len: int, side_chain_lens: List[int],
                           side_chain_residue_sequences: List[str]) -> SideChain:
        if self._is_side_chain_present(main_bead_id, side_chain_lens,
                                       side_chain_residue_sequences):
            side_chain = SideChain(main_chain_len, main_bead_id, side_chain_lens[main_bead_id],
                                   [side_chain_residue_sequences[main_bead_id]])
        else:
            side_chain = None
        return side_chain

    def _is_side_chain_present(self, main_bead_id: int, side_chain_lens: List[int],
                               side_chain_residue_sequences) -> bool:
        return side_chain_lens and side_chain_lens[
            main_bead_id] != 0 and side_chain_residue_sequences and \
               side_chain_residue_sequences[main_bead_id] is not None
