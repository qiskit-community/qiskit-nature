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
"""A class defining a main chain of a peptide."""
from typing import List, Sequence

from ...exceptions.invalid_side_chain_exception import (
    InvalidSideChainException,
)
from ...exceptions.invalid_size_exception import InvalidSizeException
from ...peptide.beads.main_bead import MainBead
from ...peptide.chains.base_chain import BaseChain
from ...peptide.chains.side_chain import SideChain


class MainChain(BaseChain):
    """A class defining a main chain of a peptide."""

    def __init__(
        self,
        main_chain_len: int,
        main_chain_residue_sequences: List[str],
        side_chain_lens: List[int],
        side_chain_residue_sequences: List[str],
    ):
        """
        Args:
            main_chain_len: Length of the main chain of a peptide.
            main_chain_residue_sequences: List of characters that define residues for a main chain.
            side_chain_lens: List of lengths of all side chains.
            side_chain_residue_sequences: List of characters that define residues for all side
            beads.
        """
        self._main_chain_residue_sequence = main_chain_residue_sequences
        beads_list = self._build_main_chain(
            main_chain_len,
            main_chain_residue_sequences,
            side_chain_lens,
            side_chain_residue_sequences,
        )
        super().__init__(beads_list)

    @property
    def main_chain_residue_sequence(self):
        """Returns a residue sequence for the main chain."""
        return self._main_chain_residue_sequence

    def _build_main_chain(
        self,
        main_chain_len: int,
        main_chain_residue_sequences: List[str],
        side_chain_lens: List[int],
        side_chain_residue_sequences: List[str],
    ) -> Sequence[MainBead]:
        """
        Creates a main chain for a given main chain length and side chain data.
        Args:
            main_chain_len: length of the main chain of a peptide.
            main_chain_residue_sequences: list of characters that define residues for a main chain.
            side_chain_lens: list of lengths of all side chains.
            side_chain_residue_sequences: list of characters that define residues for all side
            beads.

        Returns:
            main_chain: an instance of a MainChain class.
        """
        main_chain = []
        self._validate_chain_lengths(main_chain_len, side_chain_lens)
        self._validate_side_chain_index_by_lengths(side_chain_lens)
        self._validate_side_chain_index_by_residues(side_chain_residue_sequences)

        for main_bead_id in range(main_chain_len - 1):
            bead_turn_qubit_1 = self._build_turn_qubit(main_chain_len, 2 * main_bead_id)
            bead_turn_qubit_2 = self._build_turn_qubit(main_chain_len, 2 * main_bead_id + 1)
            side_chain = self._create_side_chain(
                main_bead_id, main_chain_len, side_chain_lens, side_chain_residue_sequences
            )
            main_bead = MainBead(
                main_bead_id,
                main_chain_residue_sequences[main_bead_id],
                [bead_turn_qubit_1, bead_turn_qubit_2],
                side_chain,
            )
            main_chain.append(main_bead)
        main_bead = MainBead(main_chain_len - 1, None, None, None)
        main_chain.append(main_bead)
        return main_chain

    @staticmethod
    def _validate_chain_lengths(main_chain_len: int, side_chain_lens):
        if side_chain_lens is not None and main_chain_len != len(side_chain_lens):
            raise InvalidSizeException("side_chain_lens size not equal main_chain_len")

    @staticmethod
    def _validate_side_chain_index_by_lengths(side_chain_lens: List[int]):
        if side_chain_lens is not None and (
            side_chain_lens[0] != 0 or side_chain_lens[1] != 0 or side_chain_lens[-1] != 0
        ):
            raise InvalidSideChainException(
                "First, second and last main beads are not allowed to have a side chain. Non-zero "
                "length provided for an invalid side chain."
            )

    @staticmethod
    def _validate_side_chain_index_by_residues(side_chain_residue_sequences: List[str]):
        if side_chain_residue_sequences is not None and (
            side_chain_residue_sequences[0] is not None
            or side_chain_residue_sequences[1] is not None
            or side_chain_residue_sequences[-1] is not None
        ):
            raise InvalidSideChainException(
                "First, second and last main beads are not allowed to have a side chain. Non-None "
                "residue provided for an invalid side chain"
            )

    def _create_side_chain(
        self,
        main_bead_id: int,
        main_chain_len: int,
        side_chain_lens: List[int],
        side_chain_residue_sequences: List[str],
    ) -> SideChain:
        """
        Creates a side chain for a given main bead.
        Args:
            main_bead_id: id of a main bead that will host a side chain.
            main_chain_len: length of the main chain of a peptide.
            side_chain_lens: list of lengths of all side chains.
            side_chain_residue_sequences: list of characters that define residues for all side
            beads.

        Returns:
            side_chain: an instance of a SideChain class.
        """
        if self._is_side_chain_present(main_bead_id, side_chain_lens, side_chain_residue_sequences):
            side_chain = SideChain(
                main_chain_len,
                main_bead_id,
                side_chain_lens[main_bead_id],
                [side_chain_residue_sequences[main_bead_id]],
            )
        else:
            side_chain = None
        return side_chain

    def _is_side_chain_present(
        self, main_bead_id: int, side_chain_lens: List[int], side_chain_residue_sequences: List[str]
    ) -> bool:
        """
        Returns true if a main bead of a given id hosts a side chain. Returns false otherwise.
        Args:
            main_bead_id: id of a main bead that will host a side chain.
            side_chain_lens: list of lengths of all side chains.
            side_chain_residue_sequences: list of characters that define residues for all side
                                        beads.

        Returns:
            is_side_chain_present: a boolean indicating whether a given main bead hosts a side
            chain.
        """
        is_side_chain_present = bool(
            side_chain_lens
            and side_chain_lens[main_bead_id] != 0
            and side_chain_residue_sequences
            and side_chain_residue_sequences[main_bead_id] is not None
        )
        return is_side_chain_present
