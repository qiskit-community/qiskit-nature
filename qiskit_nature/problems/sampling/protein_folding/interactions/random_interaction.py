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
"""A class defining a random interaction between beads of a peptide."""
import numpy as np
from qiskit_nature.problems.sampling.protein_folding.interactions.interaction import Interaction


class RandomInteraction(Interaction):
    def calc_energy_matrix(self, chain_len: int, sequence):  # TODO unused arg
        """
        Calculates an energy matrix for a random interaction.
        Args:
            chain_len: Length of a protein chain.

        Returns:
            pair_energies: Numpy array of pair energies for amino acids.
        """
        pair_energies = -1 - 4 * np.random.rand(chain_len + 1, 2, chain_len + 1, 2)
        return pair_energies
