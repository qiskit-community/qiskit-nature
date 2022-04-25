# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:14:02 2022

@author: 7G5381848
"""
from typing import Union, List, Tuple


class ProteinDecoder():
    
    """This class handles the decoding of the compact solution in ProteinFoldingProblem and returns the information encoded in the result about the turns
        associated to the main and side chains.
    """
    
    
    def __init__(self, best_sequence:str ,side_chain_hot_vector: List[int], unused_qubits: List[int]) -> None:
        
        self._best_sequence = best_sequence
        self._side_chain_hot_vector = side_chain_hot_vector
        self._unused_qubits = unused_qubits
        self._main_chain_lenght = len(side_chain_hot_vector)
        
    
    def _bitstring2turns(self,bitstring) -> List[int]:
        """
        Turns a bitstring encoding the shape of the molecule and decodes it.
        
        Args:
            result_bitstring: string containing the encoded information.
        Returns:
            A list of integers deccoding the bitstring.
        """
        bitstring = bitstring[::-1]
        encoding={'00':0,'01':1,'10':2,'11':3}
        lenght_turns=len(bitstring)//2       
        return [encoding[ bitstring[2 * i : 2 * (i+1)] ] for i in range(lenght_turns)]
    
    
    def _split_bitstring(self) -> Tuple[int]:
        """Returns the ammount of bits in the compact solution corresponding to each property they encode."""
        N_qbits_encoding_main_turns = 2 * (self._main_chain_lenght-3) - (5 in self._unused_qubits[:6])
        N_qbits_encoding_side_turns = 2 * sum(self._side_chain_hot_vector)
        return N_qbits_encoding_main_turns , N_qbits_encoding_side_turns
    
    def get_main_turns(self) -> List[int]:      
        """
        Returns the list of turns for the molecule corresponding to best_sequence. 
        The first element of the list corresponds to the turn of the second aminoacid in the peptide.
        Returns:
                -A list of integers representing the sequence of turns on the molecule
            
        Notes:
                -The bitsting will end in 0010 corresponding to turn1=(01) and turn2=(00)
                -If the second bead doesn't have a side chain the 6th bit can be set to 1 without loss of generality.
                In that case index (5) will belong to the list of unused qubits
                -The amount of qubits neded to encode the turns will be 2(N-3) - 1 if no side chain on second main bead
                or 2(N-3) otherwise.
        """
        
        main_turns_bitstring = self._best_sequence[-self._split_bitstring()[0]:] + "0010"
        
        if 5 in self._unused_qubits[:6]:
            main_turns_bitstring = main_turns_bitstring[:-5] + '1' + main_turns_bitstring[-5:]
            

        return self._bitstring2turns( main_turns_bitstring)



    
    def get_side_turns(self) -> List[Union[None,int]]:
        """
        Returns the list of turns from the main bead corresponding to the side chains.
        None corresponds to no side chain from that main bead.
        
        Returns:
            A list with either an number associated to a turn from the main bead or None if there is no side bead.

        """
        N,M=self._split_bitstring()
        
        side_turns_bitstring = self._best_sequence[-N-M:-N]
        side_turns = self._bitstring2turns(side_turns_bitstring)
        result = []
        for element in self._side_chain_hot_vector:
            if element == 1 :
                result += [side_turns[::-1].pop()]
            else:
                result += [None]
        return result