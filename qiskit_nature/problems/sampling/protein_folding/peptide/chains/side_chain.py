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
"""A class defining a side chain of a peptide."""
from typing import List, Union, Sequence, Optional

from ...exceptions.invalid_side_chain_exception import (
    InvalidSideChainException,
)
from ..beads.side_bead import SideBead
from .base_chain import BaseChain


class SideChain(BaseChain):
    """A class defining a side chain of a peptide."""

    def __init__(
        self,
        main_chain_len: int,
        main_bead_id: int,
        side_chain_len: int,
        side_chain_residue_sequences: List[Optional[str]],
    ):
        """
        Args:
            main_chain_len: Length of the main chain of a peptide.
            main_bead_id: Index of the main bead which the side chain is attached to.
            side_chain_len: Length of the side chains.
            side_chain_residue_sequences: List of characters that define residues for all side
                                        beads. None if a side bead does not exist.
        """
        beads_list = self._build_side_chain(
            main_chain_len, main_bead_id, side_chain_len, side_chain_residue_sequences
        )
        super().__init__(beads_list)

    def _build_side_chain(
        self,
        main_chain_len: int,
        main_bead_id: int,
        side_chain_len: int,
        side_chain_residue_sequences: List[Optional[str]],
    ) -> Union[Sequence[SideBead], None]:
        """
        Creates a side chain for a given main bead.

        Args:
            main_bead_id: id of a main bead that will host a side chain.
            main_chain_len: length of the main chain of a peptide.
            side_chain_len: length of a given side chain.
            side_chain_residue_sequences: list of characters that define residues for all side
                                        beads. None is a side bead does not exist.

        Returns:
            An instance of a SideChain class.

        Raises:
            InvalidSideChainException: if side chains of length greater than 1 provided.
        """
        if side_chain_len > 1:
            raise InvalidSideChainException(
                f"Only side chains of length 1 supported, length {side_chain_len} was given."
            )
        if side_chain_len == 0:
            return None
        side_chain = []
        for side_bead_id in range(side_chain_len):
            bead_turn_qubit_1 = self._build_turn_qubit(main_chain_len, 2 * main_bead_id)
            bead_turn_qubit_2 = self._build_turn_qubit(main_chain_len, 2 * main_bead_id + 1)
            side_bead = SideBead(
                main_bead_id,
                side_bead_id,
                side_chain_residue_sequences[side_bead_id],
                (bead_turn_qubit_1, bead_turn_qubit_2),
            )
            side_chain.append(side_bead)
        return side_chain
