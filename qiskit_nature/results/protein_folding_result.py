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
"""The protein folding result."""
from typing import Union, List, Tuple

from qiskit.opflow import PauliOp
import numpy as np
import random
import matplotlib.pyplot as plt


from qiskit_nature.results import EigenstateResult
from qiskit.algorithms import MinimumEigensolverResult


class ProteinFoldingResult(EigenstateResult):
    """
    The Protein Folding Result
    """
    def __init__(
        self,
        raw_result: MinimumEigensolverResult,
        unused_qubits: List[int],
        main_chain_aminoacid_list : List[str],
        side_chain_hot_vector : List[bool]
        
        
    ) -> None:
        super().__init__()
        self._unused_qubits = unused_qubits
        self._raw_result=raw_result
        self._best_sequence , self._amplitude_best_sequence = self._choose_best_sequence()
        self._main_chain_aminoacid_list = main_chain_aminoacid_list
        self._side_chain_hot_vector = side_chain_hot_vector
        self._main_chain_lenght = len(self._main_chain_aminoacid_list)
        
    
    
    def _choose_best_sequence(self) -> Tuple[str,float]:
        """Returns the bitstring with the biggest amplitude in the solution."""
        best_sequence = max(self._raw_result.eigenstate, key=self._raw_result.eigenstate.get)
        return (best_sequence , self.raw_result.eigenstate[best_sequence])
    
    
    @property
    def unused_qubits(self) -> List[int]:
        """Returns the list of indices for qubits in the original problem formulation that were
        removed during compression."""
        return self._unused_qubits
    
    @property
    def best_sequence(self) -> str:
        """Returns the best sequence."""
        return self._best_sequence
    
    @property
    def main_chain_aminoacid_list(self) -> str:
        """Returns the best sequence."""
        return self._main_chain_aminoacid_list  
    
    def _bitstring2turns(self,bitstring:str, compact: bool = True) -> List[int]:
        """
        Turns a bitstring encoding the shape of the molecule and decodes it.
        
        Args:
            result_bitstring: string containing the encoded information.
            compact: if True the turns are encoded in 2 bits
                      if False the turns are encoded in 4 bits

        """
        bitstring = bitstring[::-1]
        if compact:
            encoding={'00':0,'01':1,'10':2,'11':3}
            turn_lenght = 2
        else:
            encoding={'0001':0,'0010':1,'0100':2,'1000':3}
            turn_lenght=4
            
        lenght_bitstring=len(bitstring)//turn_lenght
        turns=[-1]*lenght_bitstring     
        for i in range(lenght_bitstring):
            turns[i] = encoding[ bitstring[turn_lenght * i : turn_lenght * (i+1)] ]
        
        return turns
    
    def _generate_xyzpositions(self) -> np.array:
        """
        Returns an array with the cartesian coordinates of each aminoacid in the main chain.
        
        Returns:
            An array with the cartesian coordinates of each aminoacid in the main chain.

        """
        turns = self._get_main_turns()
        
        #Corners of a cube centered at (0,0,0) forming a tetrahedron. We normalize the bond lenghts to measure 1.
        coordinates=(1. / np.sqrt(3)) * np.array([[-1,1,1],
                                                  [1,1,-1],
                                                  [-1,-1,-1],
                                                  [1,-1,1]])
        
        lenght_turns=len(turns)
        relative_positions = np.zeros((lenght_turns+1,3),dtype=float)
        
        for i in range(lenght_turns):
            relative_positions[i+1] = (-1)**i * coordinates[turns[i]]
        
        return relative_positions.cumsum(axis=0)
    
    def _generate_side_xyzpositions(self) -> List[np.array]:
        """
        Generates the xyz positions for each side chain.
        
        Returns:
            A list with the position of the side chain of each bead in the main chain in order.
            None in the ith position of the list corresponds to no side chain at that position of the main chain.

        """
        #Corners of a cube centered at (0,0,0) forming a tetrahedron. We normalize the bond lenghts to measure 1.
        coordinates=(1. / np.sqrt(3)) * np.array([[-1,1,1],
                                                  [1,1,-1],
                                                  [-1,-1,-1],
                                                  [1,-1,1]])
        side_positions = []
        
        counter=1
        
        for mainpos,sideturn in zip(self._generate_xyzpositions(),self._get_side_turns()):
            if sideturn == None:
                side_positions.append(None)
            else:
                side_positions.append(mainpos+(-1)**counter *coordinates[sideturn])
                
            counter += 1
        return side_positions
    
    
    def xyzfile(self, name : str ='default', output_data=False) -> np.array:
        """
        Creates a .xyz file and saves it in the current directory.
        
        TODO:
            Incorporate side chains into the  file. How should the format be?
        """
        data = self._generate_xyzpositions()
        data = np.column_stack([self._main_chain_aminoacid_list,data])
        if output_data:
            np.savetxt(name+'.xyz',data , delimiter=' ', fmt = '%s')
        return data
    
    
   
   
    def get_result_binary_vector(self) -> str:
        """Returns a string that encodes a solution of the ProteinFoldingProblem.
        The ProteinFoldingProblem uses a compressed optimization problem that does not match the
        number of qubits in the original objective function. This method calculates the original
        version of the solution vector. Bits that can take any value without changing the
        solution are denoted by '*'."""
        unused_qubits = self._unused_qubits
        result = []
        offset = 0
        size = len(self._best_sequence)
        for i in range(size):
            index = size - 1 - i
            while i + offset in unused_qubits:
                result.append("*")
                offset += 1
            result.append(self._best_sequence[index])

        return "".join(result[::-1])
    
    def _split_bitstring(self) -> Tuple[int]:
        """Returns the ammount of bits in the compact solution corresponding to each property they encode."""
        N_qbits_encoding_main_turns = 2 * (self._main_chain_lenght-3) - (5 in self.unused_qubits[:6])
        N_qbits_encoding_side_turns = 2 * sum(self._side_chain_hot_vector)
        return N_qbits_encoding_main_turns , N_qbits_encoding_side_turns
    
    def _get_main_turns(self) -> List[int]:      
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
        
        main_turns_bitstring = self.best_sequence[-self._split_bitstring()[0]:] + "0010"
        
        if 5 in self.unused_qubits[:6]:
            main_turns_bitstring = main_turns_bitstring[:-5] + '1' + main_turns_bitstring[-5:]
            

        return self._bitstring2turns( main_turns_bitstring)



    
    def _get_side_turns(self) -> List[Union[None,int]]:
        """
        Returns the list of turns from the main bead corresponding to the side chains.
        None corresponds to no side chain from that main bead.
        
        Returns:
            A list with either an number associated to a turn from the main bead or None if there is no side bead.

        """
        N,M=self._split_bitstring()
        
        side_turns_bitstring = self.best_sequence[-N-M:-N]
        side_turns = self._bitstring2turns(side_turns_bitstring)
        result = []
        for element in self._side_chain_hot_vector:
            if element == 1 :
                result += [side_turns[::-1].pop()]
            else:
                result += [None]
        return result
        
    
    
     
    def plotstructure(self) -> None:
        """
        Plots the molecule in 3D.
        """

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        positions = self._generate_xyzpositions()
        X, Y, Z = positions[:,0],positions[:,1],positions[:,2]
        ax.plot3D(X, Y, Z)
        ax.scatter3D(X,Y,Z,s=500)
        
        for i,side_chain in enumerate(self._generate_side_xyzpositions()):
            if type(side_chain) == np.ndarray:
                Xs,Ys,Zs = side_chain
                ax.scatter3D(Xs,Ys,Zs,s=600,c='green')
                ax.plot3D([X[i],Xs],[Y[i],Ys],[Z[i],Zs],c='green')
        
        
        boxsize = positions.max()
        ax.set_xlim(-boxsize, boxsize)
        ax.set_ylim(-boxsize, boxsize)
        ax.set_zlim(-boxsize, boxsize)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        plt.draw()
        
                           
        return

















































