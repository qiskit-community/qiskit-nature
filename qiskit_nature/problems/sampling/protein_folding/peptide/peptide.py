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
"""A class containing information about beads and chains of a protein."""
from typing import List, Sequence

from .chains.side_chain import SideChain
from .chains.main_chain import MainChain


class Peptide:
    """A class containing information about beads and chains of a protein."""

    def __init__(
        self,
        main_chain_residue_sequence: str,
        side_chain_residue_sequences: List[str],
    ):
        """
        Args:
            main_chain_residue_sequence: String of characters that define residues for a main
                                        chain. Valid residue types are [A, C, D, E, F, G, H, I,
                                        K, L, M, N, P, Q, R, S, T, V, W, Y].
            side_chain_residue_sequences: List of characters that define residues for all side
                                        beads. Empty string if a side bead does not exist. Valid
                                        residue types are [A, C, D, E, F, G, H, I, K, L, M, N, P,
                                        Q, R, S, T, V, W, Y].
        """

        self._main_chain = MainChain(
            main_chain_residue_sequence,
            side_chain_residue_sequences,
        )

    def get_side_chains(self) -> Sequence[SideChain]:
        """
        Returns the list of all side chains in a peptide.

        Returns:
            A list of all side chains in a peptide.
        """
        side_chains = []
        for main_bead in self._main_chain.beads_list:
            side_chains.append(main_bead.side_chain)  # type: ignore
        return side_chains

    def get_side_chain_hot_vector(self) -> List[bool]:
        """
        Returns a one-hot encoding list for side chains in a peptide which indicates which side
        chains are present.

        Returns:
            A one-hot encoding list for side chains in a peptide.
        """
        side_chain_hot_vector = []
        for main_bead in self._main_chain.beads_list:
            if main_bead.side_chain is not None:  # type: ignore
                side_chain_hot_vector.append(True)
            else:
                side_chain_hot_vector.append(False)
        return side_chain_hot_vector

    @property
    def get_main_chain(self) -> MainChain:
        """Returns the main chain."""
        return self._main_chain
