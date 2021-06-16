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
from typing import List, Union

from qiskit.opflow import PauliSumOp, PauliOp

from problems.sampling.protein_folding.bead_distances.distance_map_builder import \
    _create_distance_qubits
from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_full_identity
from problems.sampling.protein_folding.peptide.peptide import Peptide
from problems.sampling.protein_folding.qubit_utils.qubit_fixing import _fix_qubits


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

    def _first_neighbor(self,
                        peptide: Peptide,
                        lower_bead_ind: int,
                        is_side_chain_upper: int,
                        upper_bead_ind: int,
                        is_side_chain_lower: int,
                        lambda_1: float,
                        pair_energies: List[List[List[List[float]]]],
                        pair_energies_multiplier: float = 0.1,
                        ) -> Union[PauliSumOp, PauliOp]:
        """
        Creates first nearest neighbor interaction if beads are in contact
        and at a distance of 1 unit from each other. Otherwise, a large positive
        energetic penalty is added. Here, the penalty depends on the neighboring
        beads of interest (i and j), that is, lambda_0 > 6*(j -i + 1)*lambda_1 + e_ij.
        Here, we chose, lambda_0 = 7*(j- 1 + 1).

        Args:
            peptide: A Peptide object that includes all information about a protein.
            lower_bead_ind: Backbone bead at turn i.
            upper_bead_ind: Backbone bead at turn j (j > i).
            is_side_chain_upper: Side chain on backbone bead j.
            is_side_chain_lower: Side chain on backbone bead i.
            lambda_1: Constraint to penalize local overlap between
                     beads within a nearest neighbor contact.
            pair_energies: Numpy array of pair energies for amino acids.
            pair_energies_multiplier: A constant that multiplies pair energy contributions.

        Returns:
            expr: Contribution to an energetic Hamiltonian.
        """
        bounding_constant = 7
        lambda_0 = bounding_constant * (upper_bead_ind - lower_bead_ind + 1) * lambda_1
        lower_bead = peptide.get_main_chain[lower_bead_ind - 1]
        upper_bead = peptide.get_main_chain[upper_bead_ind - 1]
        if is_side_chain_upper == 1:
            lower_bead = lower_bead.side_chain[0]
        if is_side_chain_lower == 1:
            upper_bead = upper_bead.side_chain[0]
        energy = pair_energies[
            lower_bead_ind, is_side_chain_upper, upper_bead_ind, is_side_chain_lower]
        x = self.distance_map[lower_bead][upper_bead]
        expr = lambda_0 * (x - _build_full_identity(
            x.num_qubits)) + pair_energies_multiplier * energy * _build_full_identity(x.num_qubits)
        return _fix_qubits(expr).reduce()

    def _second_neighbor(self,
                         peptide: Peptide,
                         lower_bead_ind: int,
                         is_side_chain_upper: int,
                         upper_bead_ind: int,
                         is_side_chain_lower: int,
                         lambda_1: float,
                         pair_energies: List[List[List[List[float]]]],
                         pair_energies_multiplier: float = 0.1,
                         ) -> Union[PauliSumOp, PauliOp]:
        """
        Creates energetic interaction that penalizes local overlap between
        beads that correspond to a nearest neighbor contact or adds no net
        interaction (zero) if beads are at a distance of 2 units from each other.
        Ensure second NN does not overlap with reference point

        Args:
            peptide: A Peptide object that includes all information about a protein.
            lower_bead_ind: Backbone bead at turn i.
            upper_bead_ind: Backbone bead at turn j (j > i).
            is_side_chain_upper: Side chain on backbone bead j.
            is_side_chain_lower: Side chain on backbone bead i.
            lambda_1: Constraint to penalize local overlap between
                     beads within a nearest neighbor contact.
            pair_energies: Numpy array of pair energies for amino acids.
            pair_energies_multiplier: A constant that multiplies pair energy contributions.

        Returns:
            expr: Contribution to an energetic Hamiltonian.
        """
        energy = pair_energies[
            lower_bead_ind, is_side_chain_upper, upper_bead_ind, is_side_chain_lower]
        lower_bead = peptide.get_main_chain[lower_bead_ind - 1]
        upper_bead = peptide.get_main_chain[upper_bead_ind - 1]
        if is_side_chain_upper == 1:
            lower_bead = lower_bead.side_chain[0]
        if is_side_chain_lower == 1:
            upper_bead = upper_bead.side_chain[0]
        x = self.distance_map[lower_bead][upper_bead]
        expr = lambda_1 * (
                2 * (_build_full_identity(x.num_qubits)) - x
        ) + pair_energies_multiplier * energy * _build_full_identity(x.num_qubits)
        return _fix_qubits(expr).reduce()
