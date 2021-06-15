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
"""An abstract class defining an interaction between beads of a peptide."""
from abc import ABC, abstractmethod
from typing import List


class Interaction(ABC):
    """An abstract class defining an interaction between beads of a peptide."""

    @abstractmethod
    def calc_energy_matrix(self, chain_len: int, residue_sequence: List[str]):
        """
        Calculates an energy matrix for a particular interaction type.
        Args:
            chain_len: Length of a protein chain.
            residue_sequence: A list that contains characters defining residues for a chain of
                            proteins.

        Returns:
            pair_energies: Numpy array of pair energies for amino acids.
        """
        pass
