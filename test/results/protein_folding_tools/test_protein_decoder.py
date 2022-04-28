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

    test_decoder_1 = ProteinDecoder(best_sequence = "101100011" ,
                                    side_chain_hot_vector = [False, False, False, False, False, False, False] ,
                                    unused_qubits =[  0,   1,   2,   3,   5,  12,  13,  14,  15,  16,  17,  18,  19,
                                                     20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
                                                     33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
                                                     46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
                                                     59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
                                                     72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
                                                     85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
                                                     98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                                                    111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                                                    124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
                                                    138, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151,
                                                    152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                                                    165, 166, 167]                                    
                                    )
    
    test_decoder_2 = ProteinDecoder(best_sequence = "0011011",
                                    side_chain_hot_vector = [False, False, True, True, False] ,
                                    unused_qubits = [ 0,  1,  2,  3,  5,  8,  9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23,
                                           24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                           41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                                           58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                                           75, 76, 77, 78, 79]
                                    )
    def test_bitstring2turns(self):
        """Tests the method transforming a bitstring to an array of turns. """
        self.assertEqual(self.test_decoder_1._bitstring2turns("11001001"),[2, 1, 0, 3])
    
    def test_split_bitstring(self):
        """Tests if the bitstring is correctly split between main chain position and side chain positions. """
        self.assertEqual(self.test_decoder_1._split_bitstring(),(7, 0))
        self.assertEqual(self.test_decoder_2._split_bitstring(),(3, 4))
        
    def test_get_main_turns(self):
        """Tests the main turn list. """
        self.assertEqual(self.test_decoder_1.get_main_turns(),[1, 0, 3, 2, 0, 3])
        self.assertEqual(self.test_decoder_2.get_main_turns(),[1, 0, 3, 2])
    def test_get_side_turns(self):
        """Tests the side chain turn list"""
        self.assertEqual(self.test_decoder_1.get_side_turns(),[None, None, None, None, None, None, None])
        self.assertEqual(self.test_decoder_2.get_side_turns(),[None, None, 3, 3, None])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    