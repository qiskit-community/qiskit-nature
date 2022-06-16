# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An auxiliary class that gets the turns in the main and side chain of a molecule
 in :class:`ProteinFoldingResult`."""
from typing import List, Tuple, Optional


class ProteinShapeDecoder:
    """
    This class handles the decoding of the compact solution in
    :class:`~qiskit_nature.problems.sampling.protein_folding_problem.ProteinFoldingProblem`
    and returns the information encoded in the result about the turns
    associated to the main and side chains.
    """

    def __init__(
        self, turn_sequence: str, side_chain_hot_vector: List[bool], fifth_bit: bool
    ) -> None:
        """
        Args:
            turn_sequence: Sequence to be decoded.
            side_chain_hot_vector: A list of boolean that indicates the presence of side
                chains on corresponding indices of the main chain.
            fifth_bit: True if the 5th bit does not correspond to an unused qubit from the
                problem.
        """
        self._turn_sequence = turn_sequence
        self._side_chain_hot_vector = side_chain_hot_vector
        self._fifth_bit = fifth_bit
        self._main_chain_length = len(side_chain_hot_vector)
        self._main_turns = self._get_main_turns()
        self._side_turns = self._get_side_turns()

    @property
    def main_turns(self):
        """Returns the turns of the main chain."""
        return self._main_turns

    @property
    def side_turns(self):
        """Returns the turns of the side chain."""
        return self._side_turns

    def _bitstring_to_turns(self, bitstring: str) -> List[int]:
        """
        Takes a bitstring encoding the turns of a chain and returns the turns as a list of integers.

        Args:
            bitstring: string containing the encoded shape information.
        Returns:
            A list of integers decoding the bitstring.
        """
        bitstring = bitstring[::-1]
        encoding = {"00": 0, "01": 1, "10": 2, "11": 3}
        length_turns = len(bitstring) // 2
        return [encoding[bitstring[2 * i : 2 * (i + 1)]] for i in range(length_turns)]

    def _split_bitstring(self) -> Tuple[int, int]:
        """
        Returns the amount of bits in the compact solution corresponding
        to each property they encode.
        """
        # Let N be the length of the main chain. Usually it would have N-1 turns (each one encoded
        # with 2 bits). By symmetry we can always set the first 2 turns to an arbitrary value.
        # Therefore we only need to encode N-3 turns. By symmetry as well if we don't have a side
        # chain in the second beat we only need 1 bit to encode the third turn.
        # The final formula is # = 2*(N-3) and we subtract 1 if we don't have that secondary chain.
        n_qbits_encoding_main_turns = 2 * (self._main_chain_length - 3) - (self._fifth_bit)
        # The side chains always use 2 qubits to be encoded.
        n_qbits_encoding_side_turns = 2 * sum(self._side_chain_hot_vector)
        return n_qbits_encoding_main_turns, n_qbits_encoding_side_turns

    def _get_main_turns(self) -> List[int]:
        """
        Returns the list of turns for the molecule corresponding to its turn sequence.
        The first element of the list corresponds to the turn of the second aminoacid in the peptide.
        Returns:
                A list of integers representing the sequence of turns of the molecule.

        Notes:
                The bitstring will always end in 0010 corresponding to turn1=(01) and turn2=(00).
                If the second bead doesn't have a side chain the 6th bit
                can be set to 1 without loss of generality.
                In that case index (5) will belong to the list of unused qubits.
                The amount of qubits needed to encode the turns will be 2(N-3) - 1
                if no side chain on second main bead or 2(N-3) otherwise.
        """

        main_turns_bitstring = self._turn_sequence[-self._split_bitstring()[0] :] + "0010"

        if self._fifth_bit:
            main_turns_bitstring = main_turns_bitstring[:-5] + "1" + main_turns_bitstring[-5:]

        return self._bitstring_to_turns(main_turns_bitstring)

    def _get_side_turns(self) -> List[Optional[int]]:
        """
        Returns the list of turns from the main bead corresponding to the side chains.
        ``None`` corresponds to no side chain from that main bead.

        Returns:
            A list with either a number associated to a turn from
            the main bead or None if there is no side bead.

        """
        n, m = self._split_bitstring()

        side_turns_bitstring = self._turn_sequence[-n - m : -n]

        side_turns = self._bitstring_to_turns(side_turns_bitstring)

        result = []
        counter = 0
        for element in self._side_chain_hot_vector:
            if element:
                result.append(side_turns[counter])
                counter += 1
            else:
                result.append(None)

        return result
