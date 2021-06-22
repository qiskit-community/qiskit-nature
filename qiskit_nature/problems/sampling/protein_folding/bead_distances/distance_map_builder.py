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
from typing import Dict, DefaultDict, Tuple, Any

from qiskit.opflow import OperatorBase

from ..peptide.beads.base_bead import BaseBead
from ..qubit_utils.qubit_fixing import _fix_qubits
from ..peptide.peptide import Peptide

logger = logging.getLogger(__name__)


def _create_distance_qubits(
    peptide: Peptide,
) -> Tuple[DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]], int]:
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

    distance_map: DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]] = collections.defaultdict(
        dict
    )

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


def _calc_distances_main_chain(
    peptide: Peptide,
) -> Tuple[
    DefaultDict[Any, DefaultDict[Any, DefaultDict[Any, dict]]],
    DefaultDict[Any, DefaultDict[Any, DefaultDict[Any, dict]]],
    DefaultDict[Any, DefaultDict[Any, DefaultDict[Any, dict]]],
    DefaultDict[Any, DefaultDict[Any, DefaultDict[Any, dict]]],
]:
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


def _add_distances_side_chain(
    peptide: Peptide,
    delta_n0: DefaultDict[int, DefaultDict[int, DefaultDict[int, dict]]],
    delta_n1: DefaultDict[int, DefaultDict[int, DefaultDict[int, dict]]],
    delta_n2: DefaultDict[int, DefaultDict[int, DefaultDict[int, dict]]],
    delta_n3: DefaultDict[int, DefaultDict[int, DefaultDict[int, dict]]],
) -> Tuple[
    DefaultDict[int, DefaultDict[int, DefaultDict[int, dict]]],
    DefaultDict[int, DefaultDict[int, DefaultDict[int, dict]]],
    DefaultDict[int, DefaultDict[int, DefaultDict[int, dict]]],
    DefaultDict[int, DefaultDict[int, DefaultDict[int, dict]]],
]:
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
                    delta_n0[lower_bead_ind][0][upper_bead_ind][1] = _calc_distance_term_main_side(
                        delta_n0, indic_0, lower_bead_ind, upper_bead_ind)
                    delta_n1[lower_bead_ind][0][upper_bead_ind][1] = _calc_distance_term_main_side(
                        delta_n1, indic_1, lower_bead_ind, upper_bead_ind)
                    delta_n2[lower_bead_ind][0][upper_bead_ind][1] = _calc_distance_term_main_side(
                        delta_n2, indic_2, lower_bead_ind, upper_bead_ind)
                    delta_n3[lower_bead_ind][0][upper_bead_ind][1] = _calc_distance_term_main_side(
                        delta_n3, indic_3, lower_bead_ind, upper_bead_ind)
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
                    delta_n0[lower_bead_ind][1][upper_bead_ind][0] = _calc_distance_term_side_main(delta_n0, indic_0,
                                                                                                   lower_bead_ind,
                                                                                                   upper_bead_ind)
                    delta_n1[lower_bead_ind][1][upper_bead_ind][0] = _calc_distance_term_side_main(delta_n1, indic_1,
                                                                                                   lower_bead_ind,
                                                                                                   upper_bead_ind)
                    delta_n2[lower_bead_ind][1][upper_bead_ind][0] = _calc_distance_term_side_main(delta_n2, indic_2,
                                                                                                   lower_bead_ind,
                                                                                                   upper_bead_ind)
                    delta_n3[lower_bead_ind][1][upper_bead_ind][0] = _calc_distance_term_side_main(delta_n3, indic_3,
                                                                                                   lower_bead_ind,
                                                                                                   upper_bead_ind)
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

                    delta_n0[lower_bead_ind][1][upper_bead_ind][1] = _calc_distance_term_side_side(
                        delta_n0, higher_indic_0, lower_bead_ind, lower_indic_0, upper_bead_ind)
                    delta_n1[lower_bead_ind][1][upper_bead_ind][1] = _calc_distance_term_side_side(
                        delta_n1, higher_indic_1, lower_bead_ind, lower_indic_1, upper_bead_ind)
                    delta_n2[lower_bead_ind][1][upper_bead_ind][1] = _calc_distance_term_side_side(
                        delta_n2, higher_indic_2, lower_bead_ind, lower_indic_2, upper_bead_ind)
                    delta_n3[lower_bead_ind][1][upper_bead_ind][1] = _calc_distance_term_side_side(
                        delta_n3, higher_indic_3, lower_bead_ind, lower_indic_3, upper_bead_ind)
                except KeyError:
                    pass
    return delta_n0, delta_n1, delta_n2, delta_n3


def _calc_distance_term_side_side(delta_n0, higher_indic_fun, lower_bead_ind, lower_indic_fun,
                                  upper_bead_ind):
    return _fix_qubits(
        (
                delta_n0[lower_bead_ind][0][upper_bead_ind][0]
                + (-1) ** upper_bead_ind * higher_indic_fun
                - (-1) ** lower_bead_ind * lower_indic_fun
        )
    ).reduce()


def _calc_distance_term_main_side(delta_n0, indic_fun, lower_bead_ind, upper_bead_ind):
    return _fix_qubits(
        (
                delta_n0[lower_bead_ind][0][upper_bead_ind][0]
                + (-1) ** upper_bead_ind * indic_fun
        )
    ).reduce()


def _calc_distance_term_side_main(delta_n0, indic_fun, lower_bead_ind, upper_bead_ind):
    return _fix_qubits(
        (
                delta_n0[lower_bead_ind][0][upper_bead_ind][0]
                - (-1) ** lower_bead_ind * indic_fun
        )
    ).reduce()
