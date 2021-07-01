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
from typing import Union, Tuple
import numpy as np

from qiskit.opflow import PauliSumOp, PauliOp, OperatorBase

from .distance_map_builder import (
    _create_distance_qubits,
)
from ..peptide.beads.base_bead import BaseBead
from ..peptide.pauli_ops_builder import _build_full_identity
from ..peptide.peptide import Peptide
from ..qubit_utils.qubit_fixing import _fix_qubits


class DistanceMap:
    """Stores distances between beads of a peptide as qubit operators."""

    def __init__(self, peptide: Peptide):
        """
        Args:
            peptide: A Peptide object that includes all information about a protein.
        """
        self._peptide = peptide
        self._distance_map, self._num_distances = _create_distance_qubits(peptide)

    def __getitem__(self, position: Tuple[BaseBead, BaseBead]) -> OperatorBase:
        item1, item2 = position
        return self._distance_map[item1][item2]

    @property
    def peptide(self) -> Peptide:
        """Returns a peptide."""
        return self._peptide

    @property
    def distance_map(self):
        """Returns a distance map."""
        return self._distance_map

    @property
    def num_distances(self):
        """Returns the number of distances calculated."""
        return self._num_distances

    def _first_neighbor(
        self,
        peptide: Peptide,
        lower_bead_ind: int,
        is_side_chain_upper: int,
        upper_bead_ind: int,
        is_side_chain_lower: int,
        lambda_1: float,
        pair_energies: np.ndarray,
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
            expression: Contribution to an energetic Hamiltonian.
        """
        bounding_constant = 7
        lambda_0 = bounding_constant * (upper_bead_ind - lower_bead_ind + 1) * lambda_1
        lower_bead = peptide.get_main_chain[lower_bead_ind - 1]
        upper_bead = peptide.get_main_chain[upper_bead_ind - 1]
        if is_side_chain_upper == 1:
            lower_bead = lower_bead.side_chain[0]
        if is_side_chain_lower == 1:
            upper_bead = upper_bead.side_chain[0]
        energy = pair_energies[lower_bead_ind][is_side_chain_upper][upper_bead_ind][
            is_side_chain_lower
        ]
        x = self.distance_map[lower_bead][upper_bead]
        expression = lambda_0 * (
            x - _build_full_identity(x.num_qubits)
        ) + pair_energies_multiplier * energy * _build_full_identity(x.num_qubits)
        return _fix_qubits(expression).reduce()

    def _second_neighbor(
        self,
        peptide: Peptide,
        lower_bead_ind: int,
        is_side_chain_lower: int,
        upper_bead_ind: int,
        is_side_chain_upper: int,
        lambda_1: float,
        pair_energies: np.ndarray,
        pair_energies_multiplier: float = 0.1,
    ) -> Union[PauliSumOp, PauliOp]:
        """
        Creates energetic interaction that penalizes local overlap between
        beads that correspond to a nearest neighbor contact or adds no net
        interaction (zero) if beads are at a distance of 2 units from each other.
        Ensure second nearest neighbor does not overlap with reference point

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
            expression: Contribution to an energetic Hamiltonian.
        """
        energy = pair_energies[lower_bead_ind][is_side_chain_upper][upper_bead_ind][
            is_side_chain_lower
        ]
        lower_bead = peptide.get_main_chain[lower_bead_ind - 1]
        upper_bead = peptide.get_main_chain[upper_bead_ind - 1]
        if is_side_chain_lower == 1:
            lower_bead = lower_bead.side_chain[0]
        if is_side_chain_upper == 1:
            upper_bead = upper_bead.side_chain[0]
        x = self.distance_map[lower_bead][upper_bead]
        expression = lambda_1 * (
            2 * (_build_full_identity(x.num_qubits)) - x
        ) + pair_energies_multiplier * energy * _build_full_identity(x.num_qubits)
        return _fix_qubits(expression).reduce()
