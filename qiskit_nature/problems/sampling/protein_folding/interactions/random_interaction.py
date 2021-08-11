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
from qiskit.utils import algorithm_globals

from .interaction import Interaction


class RandomInteraction(Interaction):
    """A class defining a random interaction between beads of a peptide."""

    def calculate_energy_matrix(self, residue_sequence: str) -> np.ndarray:
        """
        Calculates an energy matrix for a random interaction.

        Args:
            residue_sequence: Dummy residue sequence; only used to infer the length of a chain.

        Returns:
            Numpy array of pair energies for amino acids.
        """
        chain_len = len(residue_sequence)
        pair_energies = -1 - 4 * algorithm_globals.random.random(
            (chain_len + 1, 2, chain_len + 1, 2)
        )
        return pair_energies
