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
from __future__ import annotations

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


import qiskit_nature.problems.sampling.protein_folding.protein_folding_problem as pfp


class ProteinFoldingResult(EigenstateResult):
    """
    The Protein Folding Result
    """
    def __init__(
        self,
        protein_folding_problem : pfp.ProteinFoldingProblem,
        best_sequence: Union[str, PauliOp],
    ) -> None:
        super().__init__()
        self._protein_folding_problem = protein_folding_problem
        self._best_sequence  = best_sequence
        self._unused_qubits = self._protein_folding_problem.unused_qubits
        self._main_chain_lenght = len(self._protein_folding_problem.peptide.get_main_chain.main_chain_residue_sequence)
        self._side_chain_hot_vector = self._protein_folding_problem.peptide.get_side_chain_hot_vector()
        
    @property
    def protein_decoder(self):
        """Returns (and generates if needed) a ProteinDecoder. This class will interpret the result bitstring and return the encoded information."""
        if not hasattr(self,'_protein_decoder'):
            self._protein_decoder = ProteinDecoder(self._best_sequence, self._side_chain_hot_vector, self._unused_qubits)
        return self._protein_decoder
    
    @property
    def protein_xyz(self):
        """Returns (and generates if needed) a ProteinXYZ. This class will take the encoded turns and generate the position of every bead in the main and side chains."""
        if not hasattr(self,'_protein_xyz'):
            self._protein_xyz = ProteinXYZ(self.protein_decoder.get_main_turns(),self.protein_decoder.get_side_turns(),self._protein_folding_problem.peptide)
        return self._protein_xyz
    
    @property
    def best_sequence(self) -> str:
        """Returns the best sequence."""
        return self._best_sequence
    
   
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
        protein_plotter = ProteinPlotter()
        protein_plotter.plot(self.protein_xyz.main_positions,self.protein_xyz.side_positions)
        return
    















































