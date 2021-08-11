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
from typing import List, Union

import numpy as np


class Interaction(ABC):
    """An abstract class defining an interaction between beads of a peptide."""

    @abstractmethod
    def calculate_energy_matrix(self, residue_sequence: str) -> np.ndarray:
        """
        Calculates an energy matrix for a particular interaction type.

        Args:
            residue_sequence: A string that contains characters defining residues for
                            a chain of proteins.

        Returns:
            Numpy array of pair energies for amino acids.
        """
        pass
