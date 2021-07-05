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
from typing import Dict, DefaultDict, Tuple, Union

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
    (
        distance_map_axis_0,
        distance_map_axis_1,
        distance_map_axis_2,
        distance_map_axis_3,
    ) = _calc_distances_main_chain(peptide)
    (
        distance_map_axis_0,
        distance_map_axis_1,
        distance_map_axis_2,
        distance_map_axis_3,
    ) = _add_distances_side_chain(
        peptide, distance_map_axis_0, distance_map_axis_1, distance_map_axis_2, distance_map_axis_3
    )
    main_chain_len = len(peptide.get_main_chain)

    num_distances = 0

    distance_map: DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]] = collections.defaultdict(
        dict
    )

    for lower_bead_ind in range(1, main_chain_len):  # upper_bead_ind>lower_bead_ind
        for upper_bead_ind in range(lower_bead_ind + 1, main_chain_len + 1):
            lower_main_bead, lower_side_bead = _get_main_and_side_beads(lower_bead_ind, peptide)
            upper_main_bead, upper_side_bead = _get_main_and_side_beads(upper_bead_ind, peptide)

            distance_map[lower_main_bead][upper_main_bead] = _calc_distance(
                distance_map_axis_0,
                distance_map_axis_1,
                distance_map_axis_2,
                distance_map_axis_3,
                lower_main_bead,
                upper_main_bead,
            )
            if distance_map[lower_main_bead][upper_main_bead] != 0:
                num_distances += 1

            distance_map[lower_side_bead][upper_main_bead] = _calc_distance(
                distance_map_axis_0,
                distance_map_axis_1,
                distance_map_axis_2,
                distance_map_axis_3,
                lower_side_bead,
                upper_main_bead,
            )
            if distance_map[lower_side_bead][upper_main_bead] != 0:
                num_distances += 1

            distance_map[lower_main_bead][upper_side_bead] = _calc_distance(
                distance_map_axis_0,
                distance_map_axis_1,
                distance_map_axis_2,
                distance_map_axis_3,
                lower_main_bead,
                upper_side_bead,
            )
            if distance_map[lower_main_bead][upper_side_bead] != 0:
                num_distances += 1

            distance_map[lower_side_bead][upper_side_bead] = _calc_distance(
                distance_map_axis_0,
                distance_map_axis_1,
                distance_map_axis_2,
                distance_map_axis_3,
                lower_side_bead,
                upper_side_bead,
            )
            if distance_map[lower_side_bead][upper_side_bead] != 0:
                num_distances += 1

    logger.info(num_distances, " distances created")
    return distance_map, num_distances


def _calc_distance(
    distance_map_axis_0,
    distance_map_axis_1,
    distance_map_axis_2,
    distance_map_axis_3,
    lower_bead,
    upper_bead,
):
    return _fix_qubits(
        (
            distance_map_axis_0[lower_bead][upper_bead] ** 2
            + distance_map_axis_1[lower_bead][upper_bead] ** 2
            + distance_map_axis_2[lower_bead][upper_bead] ** 2
            + distance_map_axis_3[lower_bead][upper_bead] ** 2
        )
    )


def _calc_distances_main_chain(
    peptide: Peptide,
) -> Tuple[
    DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
    DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
    DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
    DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
]:
    """
    Calculates distance between beads based on the number of turns in
    the main chain. Note, here we consider distances between beads
    not on side chains. For a particular axis, a, we calculate the
    distance between lower_bead_ind and upper_bead_ind bead pairs,
    distance_map_axis_a = summation (k = lower_bead_ind to upper_bead_ind - 1) of (-1)^k*indica(k).
    Args:
        peptide: A Peptide object that includes all information about a protein.

    Returns:
        distance_map_axis_0, distance_map_axis_1, distance_map_axis_2, distance_map_axis_3: Tuple
        corresponding to the number of occurrences of turns at axes 0,1,2,3.
    """
    main_chain_len = len(peptide.get_main_chain)
    (
        distance_map_axis_0,
        distance_map_axis_1,
        distance_map_axis_2,
        distance_map_axis_3,
    ) = _init_dicts()
    for lower_bead_ind in range(1, main_chain_len):
        for upper_bead_ind in range(lower_bead_ind + 1, main_chain_len + 1):
            lower_main_bead = peptide.get_main_chain[lower_bead_ind - 1]
            upper_main_bead = peptide.get_main_chain[upper_bead_ind - 1]

            for k in range(lower_bead_ind, upper_bead_ind):
                indic_0, indic_1, indic_2, indic_3 = peptide.get_main_chain[
                    k - 1
                ].get_indicator_functions()
                distance_map_axis_0[lower_main_bead][upper_main_bead] += (-1) ** k * indic_0
                distance_map_axis_1[lower_main_bead][upper_main_bead] += (-1) ** k * indic_1
                distance_map_axis_2[lower_main_bead][upper_main_bead] += (-1) ** k * indic_2
                distance_map_axis_3[lower_main_bead][upper_main_bead] += (-1) ** k * indic_3

            distance_map_axis_0[lower_main_bead][upper_main_bead] = _fix_qubits(
                distance_map_axis_0[lower_main_bead][upper_main_bead]
            )
            distance_map_axis_1[lower_main_bead][upper_main_bead] = _fix_qubits(
                distance_map_axis_1[lower_main_bead][upper_main_bead]
            )
            distance_map_axis_2[lower_main_bead][upper_main_bead] = _fix_qubits(
                distance_map_axis_2[lower_main_bead][upper_main_bead]
            )
            distance_map_axis_3[lower_main_bead][upper_main_bead] = _fix_qubits(
                distance_map_axis_3[lower_main_bead][upper_main_bead]
            )

    return distance_map_axis_0, distance_map_axis_1, distance_map_axis_2, distance_map_axis_3


def _init_dicts():
    distance_map_axis_0: DefaultDict[
        BaseBead, Dict[BaseBead, Union[OperatorBase, int]]
    ] = collections.defaultdict(lambda: collections.defaultdict(int))
    distance_map_axis_1: DefaultDict[
        BaseBead, Dict[BaseBead, Union[OperatorBase, int]]
    ] = collections.defaultdict(lambda: collections.defaultdict(int))
    distance_map_axis_2: DefaultDict[
        BaseBead, Dict[BaseBead, Union[OperatorBase, int]]
    ] = collections.defaultdict(lambda: collections.defaultdict(int))
    distance_map_axis_3: DefaultDict[
        BaseBead, Dict[BaseBead, Union[OperatorBase, int]]
    ] = collections.defaultdict(lambda: collections.defaultdict(int))
    return distance_map_axis_0, distance_map_axis_1, distance_map_axis_2, distance_map_axis_3


def _add_distances_side_chain(
    peptide: Peptide,
    distance_map_axis_0: DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
    distance_map_axis_1: DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
    distance_map_axis_2: DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
    distance_map_axis_3: DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
) -> Tuple[
    DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
    DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
    DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
    DefaultDict[BaseBead, Dict[BaseBead, OperatorBase]],
]:
    """
    Calculates distances between beads located on side chains and adds the contribution to the
    distance calculated between beads (lower_bead_ind and upper_bead_ind) on the main chain. In
    the absence of side chains, this function returns a value of 0.

    Args:
        peptide: A Peptide object that includes all information about a protein.
        distance_map_axis_0: Number of occurrences of axis 0 between beads.
        distance_map_axis_1: Number of occurrences of axis 1 between beads.
        distance_map_axis_2: Number of occurrences of axis 2 between beads.
        distance_map_axis_3: Number of occurrences of axis 3 between beads.

    Returns:
        distance_map_axis_0, distance_map_axis_1, distance_map_axis_2, distance_map_axis_3:
        Updated tuple (with added side chain contributions) that track the number of occurrences
        of turns at axes 0,1,2,3.
    """
    main_chain_len = len(peptide.get_main_chain)
    side_chain = peptide.get_side_chain_hot_vector()
    for lower_bead_ind in range(1, main_chain_len):  # upper_bead_ind>lower_bead_ind
        for upper_bead_ind in range(lower_bead_ind + 1, main_chain_len + 1):
            lower_main_bead, lower_side_bead = _get_main_and_side_beads(lower_bead_ind, peptide)
            upper_main_bead, upper_side_bead = _get_main_and_side_beads(upper_bead_ind, peptide)

            upper_indic_funs = _get_indicator_funs(peptide, side_chain, upper_bead_ind)
            lower_indic_funs = _get_indicator_funs(peptide, side_chain, lower_bead_ind)

            _calc_dists_main_side_all_axes(
                distance_map_axis_0,
                distance_map_axis_1,
                distance_map_axis_2,
                distance_map_axis_3,
                upper_indic_funs,
                lower_bead_ind,
                lower_main_bead,
                peptide,
                upper_bead_ind,
                upper_side_bead,
            )

            _calc_dists_side_main_all_axes(
                distance_map_axis_0,
                distance_map_axis_1,
                distance_map_axis_2,
                distance_map_axis_3,
                lower_bead_ind,
                lower_indic_funs,
                lower_side_bead,
                peptide,
                upper_bead_ind,
                upper_main_bead,
            )

            _calc_dists_side_side_all_axes(
                distance_map_axis_0,
                distance_map_axis_1,
                distance_map_axis_2,
                distance_map_axis_3,
                upper_indic_funs,
                lower_bead_ind,
                lower_indic_funs,
                lower_side_bead,
                peptide,
                upper_bead_ind,
                upper_side_bead,
            )

    return distance_map_axis_0, distance_map_axis_1, distance_map_axis_2, distance_map_axis_3


def _get_main_and_side_beads(bead_ind, peptide):
    main_bead = peptide.get_main_chain[bead_ind - 1]
    if main_bead.side_chain:
        side_bead = main_bead.side_chain[0]
    else:
        side_bead = None
    return main_bead, side_bead


def _get_indicator_funs(peptide, side_chain, bead_ind):
    if side_chain[bead_ind - 1]:
        indic_0, indic_1, indic_2, indic_3 = (
            peptide.get_main_chain[bead_ind - 1].side_chain[0].get_indicator_functions()
        )
    else:
        indic_0, indic_1, indic_2, indic_3 = None, None, None, None
    return indic_0, indic_1, indic_2, indic_3


def _calc_dists_side_side_all_axes(
    distance_map_axis_0,
    distance_map_axis_1,
    distance_map_axis_2,
    distance_map_axis_3,
    upper_indic_funs,
    lower_bead_ind,
    lower_indic_funs,
    lower_side_bead,
    peptide,
    upper_bead_ind,
    upper_side_bead,
):

    distance_map_axes = [
        distance_map_axis_0,
        distance_map_axis_1,
        distance_map_axis_2,
        distance_map_axis_3,
    ]
    for dist_map_ax, lower_indic_fun_x, upper_indic_fun_x in zip(
        distance_map_axes, lower_indic_funs, upper_indic_funs
    ):

        dist_map_ax[lower_side_bead][upper_side_bead] = _calc_distance_term(
            peptide,
            dist_map_ax,
            lower_bead_ind,
            upper_bead_ind,
            lower_indic_fun_x,
            upper_indic_fun_x,
        )


def _calc_dists_side_main_all_axes(
    distance_map_axis_0,
    distance_map_axis_1,
    distance_map_axis_2,
    distance_map_axis_3,
    lower_bead_ind,
    indic_funs,
    lower_side_bead,
    peptide,
    upper_bead_ind,
    upper_main_bead,
):
    distance_map_axes = [
        distance_map_axis_0,
        distance_map_axis_1,
        distance_map_axis_2,
        distance_map_axis_3,
    ]
    for dist_map_ax, indic_fun_x in zip(distance_map_axes, indic_funs):
        dist_map_ax[lower_side_bead][upper_main_bead] = _calc_distance_term(
            peptide, dist_map_ax, lower_bead_ind, upper_bead_ind, indic_fun_x, None
        )


def _calc_dists_main_side_all_axes(
    distance_map_axis_0,
    distance_map_axis_1,
    distance_map_axis_2,
    distance_map_axis_3,
    indic_funs,
    lower_bead_ind,
    lower_bead,
    peptide,
    upper_bead_ind,
    upper_bead,
):
    distance_map_axes = [
        distance_map_axis_0,
        distance_map_axis_1,
        distance_map_axis_2,
        distance_map_axis_3,
    ]
    for dist_map_ax, indic_fun_x in zip(distance_map_axes, indic_funs):

        dist_map_ax[lower_bead][upper_bead] = _calc_distance_term(
            peptide, dist_map_ax, lower_bead_ind, upper_bead_ind, None, indic_fun_x
        )


def _calc_distance_term(
    peptide: Peptide,
    distance_map_axis_x,
    lower_bead_ind,
    upper_bead_ind,
    lower_indic_fun,
    upper_indic_fun,
):
    lower_main_bead = peptide.get_main_chain[lower_bead_ind - 1]
    upper_main_bead = peptide.get_main_chain[upper_bead_ind - 1]
    result = distance_map_axis_x[lower_main_bead][upper_main_bead]
    if lower_indic_fun is not None:
        result -= (-1) ** lower_bead_ind * lower_indic_fun
    if upper_indic_fun is not None:
        result += (-1) ** upper_bead_ind * upper_indic_fun

    return _fix_qubits(result)
