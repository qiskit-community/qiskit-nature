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
from .protein_folding_tools.protein_decoder import ProteinDecoder
from .protein_folding_tools.protein_xyz import ProteinXYZ
from .protein_folding_tools.protein_plotter import ProteinPlotter


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
        self._raw_result = raw_result
        self._best_sequence  = self._choose_best_sequence()
        self._main_chain_aminoacid_list = main_chain_aminoacid_list
        self._side_chain_hot_vector = side_chain_hot_vector
        self._main_chain_lenght = len(self._main_chain_aminoacid_list)
        
        self.protein_decoder=ProteinDecoder(self._best_sequence, self._side_chain_hot_vector, self._unused_qubits)
        self.protein_xyz = ProteinXYZ(self.protein_decoder.get_main_turns(),self.protein_decoder.get_side_turns(),self._main_chain_aminoacid_list)
        self.protein_plotter = ProteinPlotter()
        
    
    
    def _choose_best_sequence(self) -> Tuple[str,float]:
        """Returns the bitstring with the biggest amplitude in the solution."""
        best_sequence = max(self._raw_result.eigenstate, key=self._raw_result.eigenstate.get)
        return best_sequence
    
    
    # @property
    # def unused_qubits(self) -> List[int]:
    #     """Returns the list of indices for qubits in the original problem formulation that were
    #     removed during compression."""
    #     return self._unused_qubits
    
    @property
    def best_sequence(self) -> str:
        """Returns the best sequence."""
        return self._best_sequence
    
    # @property
    # def main_chain_aminoacid_list(self) -> str:
    #     """Returns the best sequence."""
    #     return self._main_chain_aminoacid_list  
    
   
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
    
    def get_xyz_file(self,name : str ='default', output_data=False) -> np.array:
        return self.protein_xyz.get_xyz_file(name,output_data)
    
    def plotstructure(self) -> None:
        self.protein_plotter.plot(self.protein_xyz.main_positions,self.protein_xyz.side_positions)
    
    















































