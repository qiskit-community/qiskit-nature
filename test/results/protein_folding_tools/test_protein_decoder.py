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
from qiskit_nature.results.protein_folding_tools.protein_decoder import ProteinDecoder


class TestProteinDecoder(QiskitNatureTestCase):
    """Tests ProteinDecoder."""

    test_decoder_1 = ProteinDecoder(
        best_sequence="101100011",
        side_chain_hot_vector=[False, False, False, False, False, False, False],
        fifth_bit=True,
    )

    test_decoder_2 = ProteinDecoder(
        best_sequence="0011011",
        side_chain_hot_vector=[False, False, True, True, False],
        fifth_bit=True,
    )

    test_decoder_3 = ProteinDecoder(
        best_sequence="10110110",
        side_chain_hot_vector=[False, True, False, True, False],
        fifth_bit=False,
    )

    def test_bitstring2turns(self):
        """Tests the method transforming a bitstring to an array of turns."""
        self.assertEqual(self.test_decoder_1._bitstring2turns("11001001"), [2, 1, 0, 3])

    def test_split_bitstring(self):
        """Tests if the bitstring is correctly split between main chain position and side chain positions."""
        self.assertEqual(self.test_decoder_1._split_bitstring(), (7, 0))
        self.assertEqual(self.test_decoder_2._split_bitstring(), (3, 4))
        self.assertEqual(self.test_decoder_3._split_bitstring(), (4, 4))

    def test_main_turns(self):
        """Tests the main turn list."""
        self.assertEqual(self.test_decoder_1.main_turns(), [1, 0, 3, 2, 0, 3])
        self.assertEqual(self.test_decoder_2.main_turns(), [1, 0, 3, 2])
        self.assertEqual(self.test_decoder_3.main_turns(), [1, 0, 1, 2])

    def test_side_turns(self):
        """Tests the side chain turn list"""
        self.assertEqual(
            self.test_decoder_1.side_turns(), [None, None, None, None, None, None, None]
        )
        self.assertEqual(self.test_decoder_2.side_turns(), [None, None, 3, 3, None])
        self.assertEqual(self.test_decoder_3.side_turns(), [None, 3, None, 1, None])
