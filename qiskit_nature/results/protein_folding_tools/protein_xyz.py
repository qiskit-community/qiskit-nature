# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:43:29 2022

@author: 7G5381848
"""

from typing import Union, List, Tuple
import numpy as np

class ProteinXYZ():
    """This class handles the creation of cartesian coordinates for each aminoacid in a protein and generates a .xyz file.
        It is used by Protein_Folding_result.
    """
    
    def __init__(self,main_chain_turns:List[int],side_chain_turns:List[Union[None,int]],main_chain_aminoacid_list):
        
        self._main_chain_turns = main_chain_turns
        self._side_chain_turns = side_chain_turns
        
        self._main_chain_aminoacid_list = main_chain_aminoacid_list
        
        
        #Corners of a cube centered at (0,0,0) forming a tetrahedron. We normalize the bond lenghts to measure 1.
        self.coordinates=(1. / np.sqrt(3)) * np.array([[-1,1,1],
                                                      [1,1,-1],
                                                      [-1,-1,-1],
                                                      [1,-1,1]])
        
        self._main_positions = self._generate_main_xyz_positions()
        self._side_positions = self._generate_side_xyz_positions()
        
        
        
    @property
    def main_positions(self) -> np.array:
        """Returns the positions in the main chain."""
        return self._main_positions

        
    @property
    def side_positions(self) -> List[Union[None,np.array]]:
        """Returns the positions in the side chain."""
        return self._side_positions
    
    def _generate_main_xyz_positions(self) -> np.array:
        """
        Returns an array with the cartesian coordinates of each aminoacid in the main chain.
        
        Returns:
            An array with the cartesian coordinates of each aminoacid in the main chain.

        """
              
        lenght_turns=len(self._main_chain_turns)
        relative_positions = np.zeros((lenght_turns+1,3),dtype=float)
        
        for i in range(lenght_turns):
            relative_positions[i+1] = (-1)**i * self.coordinates[self._main_chain_turns[i]]
        
        return relative_positions.cumsum(axis=0)
    
    def _generate_side_xyz_positions(self) -> List[Union[None,np.array]]:
        """
        Generates the xyz positions for each side chain.
        
        Returns:
            A list with the position of the side chain of each bead in the main chain in order.
            None in the ith position of the list corresponds to no side chain at that position of the main chain.

        """
        side_positions = []
        counter=1
        for mainpos,sideturn in zip(self.main_positions,self._side_chain_turns):
            if sideturn == None:
                side_positions.append(None)
            else:
                side_positions.append(mainpos+(-1)**counter * self.coordinates[sideturn])
                
            counter += 1
        return side_positions
    
    
    def get_xyz_file(self, name : str, output_data:bool) -> np.array:
        """
        Creates a .xyz file and saves it in the current directory.
        
        TODO:
            Incorporate side chains into the  file. How should the format be?
        """
        data = self._generate_main_xyz_positions()
        data = np.column_stack([self._main_chain_aminoacid_list,data])
        if output_data:
            np.savetxt(name+'.xyz',data , delimiter=' ', fmt = '%s')
        return 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    