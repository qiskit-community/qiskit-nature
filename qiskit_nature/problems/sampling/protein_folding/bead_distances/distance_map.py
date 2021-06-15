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
"""A class that stores distances between beads of a peptide as qubit operators."""
from problems.sampling.protein_folding.bead_distances.distance_map_builder import (
    _create_distance_qubits,
)
from problems.sampling.protein_folding.peptide.peptide import Peptide


class DistanceMap:
    """Stores distances between beads of a peptide as qubit operators."""

    def __init__(self, peptide: Peptide):
        self._peptide = peptide
        self._distance_map, self.num_distances = _create_distance_qubits(peptide)

        """    
          Args:
                peptide: A Peptide object that includes all information about a protein.
                distance_map: A beads-indexed dictionary that stores distances between beads of a 
                                peptide as qubit operators.
                num_distances: the total number of distances.
        """

    def __getitem__(self, position):
        item1, item2 = position
        return self._distance_map[item1][item2]

    @property
    def peptide(self):
        """Returns a peptide."""
        return self._peptide

    @property
    def distance_map(self):
        """Returns a distance map."""
        return self._distance_map
