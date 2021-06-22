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
"""A class defining a mixed interaction between beads of a peptide."""
from typing import List

import numpy as np
from .interaction import Interaction
from ..residue_validator import _validate_residue_sequence
from ..data_loaders.energy_matrix_loader import (
    _load_energy_matrix_file,
)


class MixedInteraction(Interaction):
    """A class defining a mixed interaction between beads of a peptide."""

    def __init__(self, additional_energies: np.ndarray = None):
        self.additional_energies = additional_energies

        """
        Args:
            additional_energies: additional energies to be used in the construction of an energy
                                    matrix for the MixedInteraction.
        """

    def calc_energy_matrix(self, chain_len: int, residue_sequence: List[str]) -> np.ndarray:
        """
        Calculates an energy matrix for a mixed interaction based on the Miyazawa-Jernigan
        potential file.
        Args:
            chain_len: Length of a protein chain.
            residue_sequence: A list that contains characters defining residues for a chain of
            proteins.

        Returns:
            pair_energies: Numpy array of pair energies for amino acids.
        """
        _validate_residue_sequence(residue_sequence)
        mj_interaction, list_aa = _load_energy_matrix_file()
        pair_energies = np.zeros((chain_len + 1, 2, chain_len + 1, 2))
        for i in range(1, chain_len + 1):
            for j in range(i + 1, chain_len + 1):
                aa_i = list_aa.index(residue_sequence[i - 1])
                aa_j = list_aa.index(residue_sequence[j - 1])
                pair_energies[i, 0, j, 0] = mj_interaction[min(aa_i, aa_j), max(aa_i, aa_j)]
            if self.additional_energies is not None:
                for interaction in self.additional_energies:
                    bead_1, bead_2, ener = tuple(interaction)
                    pair_energies[bead_1[0], bead_1[1], bead_2[0], bead_2[1]] = ener
        return pair_energies
