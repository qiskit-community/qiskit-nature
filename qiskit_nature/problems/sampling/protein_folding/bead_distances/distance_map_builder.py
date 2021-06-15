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
"""Builds a distance map that stores distances between beads in a peptide."""
import collections
import logging
from typing import List, Union

from qiskit.opflow import PauliSumOp, PauliOp

from problems.sampling.protein_folding.bead_distances.distance_map import DistanceMap
from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_full_identity
from problems.sampling.protein_folding.qubit_utils.qubit_fixing import _fix_qubits
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide

logger = logging.getLogger(__name__)


# TODO refactor the data structure storing distances
def _create_distance_qubits(peptide: Peptide):
    """
    Creates total distances between all bead pairs by summing the
    distances over all turns with axes, a = 0,1,2,3.

    Args:
        peptide: A Peptide object that includes all information about a protein.

    Returns:
        distance_map: A beads-indexed dictionary that stores distances between beads of a
                        peptide as qubit operators.
        num_distances: number of distances calculated.
    """
    delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
    delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(
        peptide, delta_n0, delta_n1, delta_n2, delta_n3
    )
    main_chain_len = len(peptide.get_main_chain)

    num_distances = 0

    distance_map = collections.defaultdict(dict)

    for lower_bead_ind in range(1, main_chain_len):  # upper_bead_ind>lower_bead_ind
        for upper_bead_ind in range(lower_bead_ind + 1, main_chain_len + 1):
            lower_main_bead = peptide.get_main_chain[lower_bead_ind - 1]
            if lower_main_bead.side_chain:
                lower_side_bead = lower_main_bead.side_chain[0]
            else:
                lower_side_bead = None
            upper_main_bead = peptide.get_main_chain[upper_bead_ind - 1]
            if upper_main_bead.side_chain:
                upper_side_bead = upper_main_bead.side_chain[0]
            else:
                upper_side_bead = None
            try:
                distance_map[lower_main_bead][upper_main_bead] = _fix_qubits(
                    (
                        delta_n0[lower_bead_ind][0][upper_bead_ind][0] ** 2
                        + delta_n1[lower_bead_ind][0][upper_bead_ind][0] ** 2
                        + delta_n2[lower_bead_ind][0][upper_bead_ind][0] ** 2
                        + delta_n3[lower_bead_ind][0][upper_bead_ind][0] ** 2
                    )
                ).reduce()
                num_distances += 1
            except KeyError:
                pass
            try:
                distance_map[lower_side_bead][upper_main_bead] = _fix_qubits(
                    (
                        delta_n0[lower_bead_ind][1][upper_bead_ind][0] ** 2
                        + delta_n1[lower_bead_ind][1][upper_bead_ind][0] ** 2
                        + delta_n2[lower_bead_ind][1][upper_bead_ind][0] ** 2
                        + delta_n3[lower_bead_ind][1][upper_bead_ind][0] ** 2
                    )
                ).reduce()
                num_distances += 1
            except KeyError:
                pass
            try:
                distance_map[lower_main_bead][upper_side_bead] = _fix_qubits(
                    (
                        delta_n0[lower_bead_ind][0][upper_bead_ind][1] ** 2
                        + delta_n1[lower_bead_ind][0][upper_bead_ind][1] ** 2
                        + delta_n2[lower_bead_ind][0][upper_bead_ind][1] ** 2
                        + delta_n3[lower_bead_ind][0][upper_bead_ind][1] ** 2
                    )
                ).reduce()
                num_distances += 1
            except KeyError:
                pass
            try:
                distance_map[lower_side_bead][upper_side_bead] = _fix_qubits(
                    (
                        delta_n0[lower_bead_ind][1][upper_bead_ind][1] ** 2
                        + delta_n1[lower_bead_ind][1][upper_bead_ind][1] ** 2
                        + delta_n2[lower_bead_ind][1][upper_bead_ind][1] ** 2
                        + delta_n3[lower_bead_ind][1][upper_bead_ind][1] ** 2
                    )
                ).reduce()
                num_distances += 1
            except KeyError:
                pass

    logger.info(num_distances, " distances created")
    return distance_map, num_distances


def _calc_distances_main_chain(peptide: Peptide):
    """
    Calculates distance between beads based on the number of turns in
    the main chain. Note, here we consider distances between beads
    not on side chains. For a particular axis, a, we calculate the
    distance between lower_bead_ind and upper_bead_ind bead pairs,
    delta_na = summation (k = lower_bead_ind to upper_bead_ind - 1) of (-1)^k*indica(k).
    Args:
        peptide: A Peptide object that includes all information about a protein.

    Returns:
        delta_n0, delta_n1, delta_n2, delta_n3: Tuple corresponding to
                                                the number of occurrences
                                                of turns at axes 0,1,2,3.
    """
    main_chain_len = len(peptide.get_main_chain)
    delta_n0, delta_n1, delta_n2, delta_n3 = (
        _init_distance_dict(),
        _init_distance_dict(),
        _init_distance_dict(),
        _init_distance_dict(),
    )
    for lower_bead_ind in range(1, main_chain_len):
        for upper_bead_ind in range(lower_bead_ind + 1, main_chain_len + 1):
            delta_n0[lower_bead_ind][0][upper_bead_ind][0] = 0
            delta_n1[lower_bead_ind][0][upper_bead_ind][0] = 0
            delta_n2[lower_bead_ind][0][upper_bead_ind][0] = 0
            delta_n3[lower_bead_ind][0][upper_bead_ind][0] = 0
            for k in range(lower_bead_ind, upper_bead_ind):
                indic_0, indic_1, indic_2, indic_3 = peptide.get_main_chain[
                    k - 1
                ].get_indicator_functions()
                delta_n0[lower_bead_ind][0][upper_bead_ind][0] += (-1) ** k * indic_0
                delta_n1[lower_bead_ind][0][upper_bead_ind][0] += (-1) ** k * indic_1
                delta_n2[lower_bead_ind][0][upper_bead_ind][0] += (-1) ** k * indic_2
                delta_n3[lower_bead_ind][0][upper_bead_ind][0] += (-1) ** k * indic_3
            delta_n0[lower_bead_ind][0][upper_bead_ind][0] = _fix_qubits(
                delta_n0[lower_bead_ind][0][upper_bead_ind][0]
            ).reduce()
            delta_n1[lower_bead_ind][0][upper_bead_ind][0] = _fix_qubits(
                delta_n1[lower_bead_ind][0][upper_bead_ind][0]
            ).reduce()
            delta_n2[lower_bead_ind][0][upper_bead_ind][0] = _fix_qubits(
                delta_n2[lower_bead_ind][0][upper_bead_ind][0]
            ).reduce()
            delta_n3[lower_bead_ind][0][upper_bead_ind][0] = _fix_qubits(
                delta_n3[lower_bead_ind][0][upper_bead_ind][0]
            ).reduce()

    return delta_n0, delta_n1, delta_n2, delta_n3


def _init_distance_dict():
    return collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(dict))
    )


def _add_distances_side_chain(peptide: Peptide, delta_n0, delta_n1, delta_n2, delta_n3):
    """
    Calculates distances between beads located on side chains and adds the contribution to the
    distance calculated between beads (lower_bead_ind and upper_bead_ind) on the main chain. In
    the absence
    of side chains, this function returns a value of 0.

    Args:
        peptide: A Peptide object that includes all information about a protein.
        delta_n0: Number of occurrences of axis 0 between beads.
        delta_n1: Number of occurrences of axis 1 between beads.
        delta_n2: Number of occurrences of axis 2 between beads.
        delta_n3: Number of occurrences of axis 3 between beads.

    Returns:
        delta_n0, delta_n1, delta_n2, delta_n3: Updated tuple (with added side chain
                                                contributions) that track the number
                                                of occurrences of turns at axes 0,1,2,3.
    """
    main_chain_len = len(peptide.get_main_chain)
    side_chain = peptide.get_side_chain_hot_vector()
    for lower_bead_ind in range(1, main_chain_len):  # upper_bead_ind>lower_bead_ind
        for upper_bead_ind in range(lower_bead_ind + 1, main_chain_len + 1):

            if side_chain[upper_bead_ind - 1]:
                try:
                    # TODO generalize to side chains longer than 1
                    indic_0, indic_1, indic_2, indic_3 = (
                        peptide.get_main_chain[upper_bead_ind - 1]
                        .side_chain[0]
                        .get_indicator_functions()
                    )
                    delta_n0[lower_bead_ind][0][upper_bead_ind][1] = _fix_qubits(
                        (
                            delta_n0[lower_bead_ind][0][upper_bead_ind][0]
                            + (-1) ** upper_bead_ind * indic_0
                        )
                    ).reduce()
                    delta_n1[lower_bead_ind][0][upper_bead_ind][1] = _fix_qubits(
                        (
                            delta_n1[lower_bead_ind][0][upper_bead_ind][0]
                            + (-1) ** upper_bead_ind * indic_1
                        )
                    ).reduce()
                    delta_n2[lower_bead_ind][0][upper_bead_ind][1] = _fix_qubits(
                        (
                            delta_n2[lower_bead_ind][0][upper_bead_ind][0]
                            + (-1) ** upper_bead_ind * indic_2
                        )
                    ).reduce()
                    delta_n3[lower_bead_ind][0][upper_bead_ind][1] = _fix_qubits(
                        (
                            delta_n3[lower_bead_ind][0][upper_bead_ind][0]
                            + (-1) ** upper_bead_ind * indic_3
                        )
                    ).reduce()
                except KeyError:
                    pass

            if side_chain[lower_bead_ind - 1]:
                try:
                    # TODO generalize to side chains longer than 1
                    indic_0, indic_1, indic_2, indic_3 = (
                        peptide.get_main_chain[lower_bead_ind - 1]
                        .side_chain[0]
                        .get_indicator_functions()
                    )
                    delta_n0[lower_bead_ind][1][upper_bead_ind][0] = _fix_qubits(
                        (
                            delta_n0[lower_bead_ind][0][upper_bead_ind][0]
                            - (-1) ** lower_bead_ind * indic_0
                        )
                    ).reduce()
                    delta_n1[lower_bead_ind][1][upper_bead_ind][0] = _fix_qubits(
                        (
                            delta_n1[lower_bead_ind][0][upper_bead_ind][0]
                            - (-1) ** lower_bead_ind * indic_1
                        )
                    ).reduce()
                    delta_n2[lower_bead_ind][1][upper_bead_ind][0] = _fix_qubits(
                        (
                            delta_n2[lower_bead_ind][0][upper_bead_ind][0]
                            - (-1) ** lower_bead_ind * indic_2
                        )
                    ).reduce()
                    delta_n3[lower_bead_ind][1][upper_bead_ind][0] = _fix_qubits(
                        (
                            delta_n3[lower_bead_ind][0][upper_bead_ind][0]
                            - (-1) ** lower_bead_ind * indic_3
                        )
                    ).reduce()
                except KeyError:
                    pass

            if side_chain[lower_bead_ind - 1] and side_chain[upper_bead_ind - 1]:
                try:
                    # TODO generalize to side chains longer than 1
                    higher_indic_0, higher_indic_1, higher_indic_2, higher_indic_3 = (
                        peptide.get_main_chain[upper_bead_ind - 1]
                        .side_chain[0]
                        .get_indicator_functions()
                    )
                    # TODO generalize to side chains longer than 1
                    lower_indic_0, lower_indic_1, lower_indic_2, lower_indic_3 = (
                        peptide.get_main_chain[lower_bead_ind - 1]
                        .side_chain[0]
                        .get_indicator_functions()
                    )

                    delta_n0[lower_bead_ind][1][upper_bead_ind][1] = _fix_qubits(
                        (
                            delta_n0[lower_bead_ind][0][upper_bead_ind][0]
                            + (-1) ** upper_bead_ind * higher_indic_0
                            - (-1) ** lower_bead_ind * lower_indic_0
                        )
                    ).reduce()
                    delta_n1[lower_bead_ind][1][upper_bead_ind][1] = _fix_qubits(
                        (
                            delta_n1[lower_bead_ind][0][upper_bead_ind][0]
                            + (-1) ** upper_bead_ind * higher_indic_1
                            - (-1) ** lower_bead_ind * lower_indic_1
                        )
                    ).reduce()
                    delta_n2[lower_bead_ind][1][upper_bead_ind][1] = _fix_qubits(
                        (
                            delta_n2[lower_bead_ind][0][upper_bead_ind][0]
                            + (-1) ** upper_bead_ind * higher_indic_2
                            - (-1) ** lower_bead_ind * lower_indic_2
                        )
                    ).reduce()
                    delta_n3[lower_bead_ind][1][upper_bead_ind][1] = _fix_qubits(
                        (
                            delta_n3[lower_bead_ind][0][upper_bead_ind][0]
                            + (-1) ** upper_bead_ind * higher_indic_3
                            - (-1) ** lower_bead_ind * lower_indic_3
                        )
                    ).reduce()
                except KeyError:
                    pass
    return delta_n0, delta_n1, delta_n2, delta_n3


def _first_neighbor(
    peptide: Peptide,
    lower_bead_ind: int,
    is_side_chain_upper: int,
    upper_bead_ind: int,
    is_side_chain_lower: int,
    lambda_1: float,
    pair_energies: List[List[List[List[float]]]],
    distance_map: DistanceMap,
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
        distance_map: A beads-indexed dictionary that stores distances between beads of a
                        peptide as qubit operators.
        pair_energies_multiplier: A constant that multiplies pair energy contributions.

    Returns:
        expr: Contribution to energetic Hamiltonian.
    """
    bounding_constant = 7
    lambda_0 = bounding_constant * (upper_bead_ind - lower_bead_ind + 1) * lambda_1
    lower_bead = peptide.get_main_chain[lower_bead_ind - 1]
    upper_bead = peptide.get_main_chain[upper_bead_ind - 1]
    if is_side_chain_upper == 1:
        lower_bead = lower_bead.side_chain[0]
    if is_side_chain_lower == 1:
        upper_bead = upper_bead.side_chain[0]
    energy = pair_energies[lower_bead_ind, is_side_chain_upper, upper_bead_ind, is_side_chain_lower]
    x = distance_map[lower_bead, upper_bead]
    expr = lambda_0 * (x - _build_full_identity(x.num_qubits))
    # + pair_energies_multiplier*energy*_build_full_identity(x.num_qubits)
    return _fix_qubits(expr).reduce()


def _second_neighbor(
    peptide: Peptide,
    lower_bead_ind: int,
    is_side_chain_upper: int,
    upper_bead_ind: int,
    is_side_chain_lower: int,
    lambda_1: float,
    pair_energies: List[List[List[List[float]]]],
    distance_map: DistanceMap,
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
        distance_map: A beads-indexed dictionary that stores distances between beads of a
                        peptide as qubit operators.
        pair_energies_multiplier: A constant that multiplies pair energy contributions.

    Returns:
        expr: Contribution to energetic Hamiltonian.
    """
    energy = pair_energies[lower_bead_ind, is_side_chain_upper, upper_bead_ind, is_side_chain_lower]
    lower_bead = peptide.get_main_chain[lower_bead_ind - 1]
    upper_bead = peptide.get_main_chain[upper_bead_ind - 1]
    if is_side_chain_upper == 1:
        lower_bead = lower_bead.side_chain[0]
    if is_side_chain_lower == 1:
        upper_bead = upper_bead.side_chain[0]
    x = distance_map[lower_bead, upper_bead]
    expr = lambda_1 * (
        2 * (_build_full_identity(x.num_qubits)) - x
    )  # + pair_energies_multiplier * energy * _build_full_identity(x.num_qubits)
    return _fix_qubits(expr).reduce()
