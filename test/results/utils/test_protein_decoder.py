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
"""Tests ProteinDecoder."""
from test import QiskitNatureTestCase
from ddt import ddt, data, unpack
from qiskit_nature.results.utils.protein_decoder import ProteinDecoder


@ddt
class TestProteinDecoder(QiskitNatureTestCase):
    """Tests ProteinDecoder."""

    @unpack
    @data(
        (
            "101100011",
            [False, False, False, False, False, False, False],
            True,
            (7, 0),
            [1, 0, 3, 2, 0, 3],
            [None, None, None, None, None, None, None],
        ),
        (
            "0011011",
            [False, False, True, True, False],
            True,
            (3, 4),
            [1, 0, 3, 2],
            [None, None, 3, 0, None],
        ),
        (
            "10110110",
            [False, True, False, True, False],
            False,
            (4, 4),
            [1, 0, 1, 2],
            [None, 3, None, 1, None],
        ),
    )
    def test_decoder(
        self, turns_sequence, side_chain_hot_vector, fifth_bit, split, main_turns, side_turns
    ):
        """
        Tests if the main and side turns are generated correctly and if the separation
        between the bits encoding side turns and main turns is correct.
        """
        decoder = ProteinDecoder(
            turns_sequence=turns_sequence,
            side_chain_hot_vector=side_chain_hot_vector,
            fifth_bit=fifth_bit,
        )
        with self.subTest("Split Bitstring"):
            self.assertEqual(decoder._split_bitstring(), split)
        with self.subTest("Main Turns"):
            self.assertEqual(decoder.main_turns, main_turns)
        with self.subTest("Side Turns"):
            self.assertEqual(decoder.side_turns, side_turns)

    def test_bitstring2turns(self):
        """Tests the method transforming a bitstring to an array of turns."""
        decoder = ProteinDecoder(
            turns_sequence="101100011",
            side_chain_hot_vector=[False, False, False, False, False, False, False],
            fifth_bit=True,
        )
        self.assertEqual(decoder._bitstring_to_turns("11001001"), [2, 1, 0, 3])
