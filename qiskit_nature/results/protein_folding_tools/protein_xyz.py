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
    
    #Corners of a cube centered at (0,0,0) forming a tetrahedron. We normalize the bond lenghts to measure 1.
    coordinates=(1. / np.sqrt(3)) * np.array([[-1,1,1],
                                              [1,1,-1],
                                              [-1,-1,-1],
                                              [1,-1,1]])
    
    def __init__(self,main_chain_turns:List[int],side_chain_turns:List[Union[None,int]],peptide) -> None:
        """
        Args:
            main_chain_turns : a list of integers encoding the turns of the main chain
            side_chain_turns : a list of integers and None encoding the turns of the main chain or None for an abscence of side chain
            peptide :  the peptide we are getting the positions for

        """
        self._main_chain_turns = main_chain_turns
        self._side_chain_turns = side_chain_turns
        
        self._main_chain_aminoacid_list = np.array( list(peptide.get_main_chain.main_chain_residue_sequence) )
        self._side_chain_aminoacid_list = np.array( [aminoacid.residue_sequence[0] if (type(aminoacid) != type(None) ) else None for aminoacid in peptide.get_side_chains()  ])


        
    @property
    def side_positions(self) -> List[Union[None,np.array]]:
        
        """
        Returns the xyz position for each side chain element. The first time called it generates the coordinates.
        
        Returns:
            A list with the position of the side chain of each bead in the main chain in order.
            None in the ith position of the list corresponds to no side chain at that position of the main chain.

        """
        if not hasattr(self,'_side_positions'):
            self._side_positions = []
            counter=1
            for mainpos,sideturn in zip(self.main_positions,self._side_chain_turns):
                if sideturn == None:
                    self._side_positions.append(None)
                else:
                    self._side_positions.append(mainpos+(-1)**counter * self.coordinates[sideturn])
                    
                counter += 1
   
        
        return self._side_positions
    
    @property
    def main_positions(self) -> np.array:
        """
        Returns an array with the cartesian coordinates of each aminoacid in the main chain. The first time called it generates the coordinates.
        
        Returns:
            An array with the cartesian coordinates of each aminoacid in the main chain.

        """
        if not hasattr(self,'_main_positions'):
            lenght_turns=len(self._main_chain_turns)
            relative_positions = np.zeros((lenght_turns+1,3),dtype=float)
            
            for i in range(lenght_turns):
                relative_positions[i+1] = (-1)**i * self.coordinates[self._main_chain_turns[i]]
            
            self._main_positions = relative_positions.cumsum(axis=0)
            
        return self._main_positions
    
    
    def get_xyz_file(self, name : str, output_data:bool) -> np.array:
        """
        Creates a .xyz file and saves it in the current directory.
        
        TODO:
            Incorporate side chains into the  file. How should the format be?
        """
        main_data = np.column_stack([self._main_chain_aminoacid_list,self.main_positions])

        #We will discard the None values corresponding to emtpy side chains.
        side_data = np.column_stack(
                                    [self._side_chain_aminoacid_list[self._side_chain_aminoacid_list != None],
                                     [side_pos for side_pos in self.side_positions if type(side_pos) != type(None)]
                                     ]) 
        if side_data.size != 0:
        
            data = np.append(main_data , side_data , axis=0)    
            
        else:
            data=main_data
        
        if output_data:
            np.savetxt(name+'.xyz',data , delimiter=' ', fmt = '%s')
        
        return data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    