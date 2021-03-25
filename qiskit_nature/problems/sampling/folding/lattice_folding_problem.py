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

"""The Molecular Problem class."""
from typing import List, Tuple, Optional
import numpy as np

from .folding_qubit_op_builder import *

class LatticeFoldingProblem:
    """Lattice Folding problem for a N-letter peptide"""
    def __init__(self,
                 residue_sequence = 'APRLRFY',
                 interaction_type = 'MJ',
                 additional_energies = [],
                 list_side_chains = [],
                 lambda_chiral = 10,
                 lambda_back = 10,
                 lambda_1 = 10,
                 lambda_contacts = 0,
                 N_contacts = 0) -> None:
        """
        Args:
            residue_sequence: Letter sequence of peptide to be analyzed
            interaction_type: Type of pairwise interaction(1-NN) between amino acids,
                              can be ``random``, ``mix`` or ``mj``, that is, from 
                              Miyazawa and Jernigan (MJ) potential 
            additional_energies: Additional energies for pairwise interactions
            list_side_chains: List of side chains in peptide
            lambda_chiral: Chirality constraint/penality that enforces the correct
                           stereochemistry if side chains are present
            lambda_back: Constraint/penalty that prevents the chain from folding
                         back onto itself
            lambda_1: 
            lambda_contacts:
            N_contacts:

        Raises:
        """

        self._residue_sequence = residue_sequence
        self._num_beads = len(residue_sequence)
        self._interaction_type = interaction_type
        self._additional_energies = additional_energies
        self._lambda_chiral = lambda_chiral
        self._lambda_back = lambda_back
        self._lambda_1 = lambda_1
        self._lambda_contacts = lambda_contacts
        self._N_contacts = N_contacts
        self._pair_energies = np.zeros((num_beads, 2, num_beads, 2))
        self._path = "./mj_matrix"

        if interaction_type == 'MJ'or interaction_type == 'mix':
            self.sequence = list(residue_sequence)
            side_chains = [0]*self._num_beads
            for s in list_side_chains:
                side_chains[s-1]=1
            self._side_chains = side_chains
        elif interaction_type == 'random':
            if 1 in list_side_chains or 2 in list_side_chains or self._num_beads in list_side_chains:
                raise Exception('No side chain on residues 1, 2 or N allowed')
            side_chains = [0]*self._num_beads
            for s in list_side_chains:
                side_chains[s-1]=1
            self._side_chains = side_chains
            self.sequence = 0

    def _load_energy_matrix_file(self):
        """Returns the energy matrix from the MJ potential file"""
        matrix = np.loadtxt(fname=self._path, dtype=str)
        MJ = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]))
        for i in range(1, np.shape(matrix)[0]):
            for j in range(i-1, np.shape(matrix)[1]):
                MJ[i, j] = float(matrix[i, j])
        MJ = MJ[1:, ]
        list_AA = list(matrix[0, :])
        # return MJ, list_AA
        self._MJ = MJ
        self._list_AA = list_AA

    def _construct_specific_pair_energy_matrix(self):
        """
        Constructs the energy matrix that describe pairwise (1-NN) 
        interaction energies between amino acids
        
        Raises:
            Exception: If interaction type is not defined.
        """
        if self._interaction_type == 'random':
            self._pair_energies = - 1 - 4*np.random.rand(self._num_beads + 1, 2, self._num_beads + 1, 2)
        elif self._interaction_type == 'MJ' or self._interaction_type == 'mix':
            self._load_energy_matrix_file()
            self._pair_energies = np.zeros((self._num_beads + 1, 2, self._num_beads + 1, 2))
            for i in range(1, self._num_beads + 1):
                for j in range(i + 1, self._num_beads + 1):
                    aa_i = self._list_AA.index(self.sequence[i-1])
                    aa_j = self._list_AA.index(self.sequence[j-1])
                    self._pair_energies[i,0,j,0] = self._MJ[min(aa_i, aa_j), max(aa_i, aa_j)]
            if self._interaction_type == 'mix':
                for interaction in self._additional_energies:
                    b1,  b2,  ener = tuple(interaction)
                    self._pair_energies[b1[0],b1[1],b2[0],b2[1]] = ener
        else:
            raise Exception('choose random or default interaction type')

    def pauli_op(self):
        """Get the qubit operator from the builder"""

        operator = _build_qubit_op(self._num_beads, self._side_chains, self._pair_energies, 
                                   self._lambda_chiral, self._lambda_back, self._lambda_1,
                                   self._lambda_contacts, self._N_contacts)
        return operator





