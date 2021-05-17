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

    def __init__(self, main_chain_len, main_chain_residue_seq, side_chain_lens: List[int],
                 side_chain_residue_sequences: str):
        beads_list = self._build_main_chain(main_chain_len, main_chain_residue_seq, side_chain_lens,
                                            side_chain_residue_sequences)
        super().__init__(beads_list)

    def _build_main_chain(self, main_chain_len, main_chain_residue_seq, side_chain_lens,
                          side_chain_residue_sequences) -> List[MainBead]:
        main_chain = []
        if side_chain_lens is not None and main_chain_len != len(side_chain_lens):
            raise InvalidSizeException("side_chain_lens size not equal main_chain_len")
        if side_chain_lens is not None and (side_chain_lens[0] != 0 or side_chain_lens[1] != 0):
            raise InvalidSideChainException(
                "First and second main beads are not allowed to have a side chain. Non-zero "
                "length provided for an inalid side chain")
        if side_chain_residue_sequences is not None and (
                side_chain_residue_sequences[0] is not None or side_chain_residue_sequences[
            1] is not None):
            raise InvalidSideChainException(
                "First and second main beads are not allowed to have a side chain. Non-None "
                "residue provided for an invalid side chain")
        for main_bead_id in range(main_chain_len):
            bead_turn_qubit_1 = self._build_turn_qubit(main_chain_len, main_bead_id)
            bead_turn_qubit_2 = self._build_turn_qubit(main_chain_len, main_bead_id + 1)
            if side_chain_lens and side_chain_lens[
                main_bead_id] != 0 and side_chain_residue_sequences and \
                    side_chain_residue_sequences[main_bead_id] is not None:
                side_chain = SideChain(side_chain_lens[main_bead_id], side_chain_residue_sequences)
            else:
                side_chain = None
            main_bead = MainBead(main_chain_residue_seq[main_bead_id],
                                 [bead_turn_qubit_1, bead_turn_qubit_2],
                                 side_chain)
            main_chain.append(main_bead)
        return main_chain
