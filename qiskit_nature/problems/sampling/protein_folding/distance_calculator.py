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
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide


# TODO refactor the data structure storing distances
def _calc_total_distances(peptide, delta_n0, delta_n1,
                          delta_n2, delta_n3):
    """
    Creates total distances between all bead pairs by summing the
    distances over all turns with axes, a = 0,1,2,3. For bead i with
    side chain s and bead j with side chain p, where j > i, the distance
    can be referenced as x_dist[i][p][j][s]

    Args:
        delta_n0: Number of occurrences of axis 0 between beads
        delta_n1: Number of occurrences of axis 1 between beads
        delta_n2: Number of occurrences of axis 2 between beads
        delta_n3: Number of occurrences of axis 3 between beads
        pauli_conf: Dictionary of conformation Pauli operators in symbolic notation

    Returns:
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3
    """
    main_chain_len = len(peptide.get_main_chain)
    # initializes dictionaries
    x_dist = dict()
    r = 0
    for i in range(1, main_chain_len):
        x_dist[i] = dict()
        x_dist[i][0], x_dist[i][1] = dict(), dict()
        for j in range(i + 1, main_chain_len + 1):
            x_dist[i][0][j], x_dist[i][1][j] = dict(), dict()

    for i in range(1, main_chain_len):  # j>i
        for j in range(i + 1, main_chain_len + 1):
            for s in range(2):  # side chain on bead i
                for p in range(2):  # side chain on bead j
                    if i == 1 and p == 1 or j == main_chain_len and s == 1:
                        continue
                    try:
                        x_dist[i][p][j][s] = (delta_n0[i][p][j][s] ** 2 +
                                              delta_n1[i][p][j][s] ** 2 +
                                              delta_n2[i][p][j][s] ** 2 +
                                              delta_n3[i][p][j][s] ** 2).reduce()
                        r += 1
                    except:
                        pass
    print(r, ' distances created')
    return x_dist


def _calc_distances_main_chain(peptide: Peptide):
    """
    Calculates distance between beads based on the number of turns in
    the main chain. Note, here we consider distances between beads
    not on side chains. For a particular axis, a, we calculate the
    distance between i and j bead pairs,
    delta_na = summation (k = i to j - 1) of (-1)^k*indica(k)
    Args:
        main_chain_len: Number of total beads in peptide
        indic_0: Turn indicator for axis 0
        indic_1: Turn indicator for axis 1
        indic_2: Turn indicator for axis 2
        indic_3: Turn indicator for axis 3
    Returns:
        delta_n0, delta_n1, delta_n2, delta_n3: Tuple corresponding to
                                                the number of occurrences
                                                of turns at axes 0,1,2,3
    """
    main_chain_len = len(peptide.get_main_chain)
    delta_n0, delta_n1, delta_n2, delta_n3 = _init_distance_dict(main_chain_len)
    # calculate distances
    for i in range(1, main_chain_len):  # j>i
        for j in range(i + 1, main_chain_len + 1):
            delta_n0[i][0][j][0] = 0
            delta_n1[i][0][j][0] = 0
            delta_n2[i][0][j][0] = 0
            delta_n3[i][0][j][0] = 0
            for k in range(i, j):
                indic_0, indic_1, indic_2, indic_3 = peptide.get_main_chain[k].get_indicator_functions()
                delta_n0[i][0][j][0] += (-1) ** k * indic_0
                delta_n1[i][0][j][0] += (-1) ** k * indic_1
                delta_n2[i][0][j][0] += (-1) ** k * indic_2
                delta_n3[i][0][j][0] += (-1) ** k * indic_3
            delta_n0[i][0][j][0] = delta_n0[i][0][j][0].reduce()
            delta_n1[i][0][j][0] = delta_n1[i][0][j][0].reduce()
            delta_n2[i][0][j][0] = delta_n2[i][0][j][0].reduce()
            delta_n3[i][0][j][0] = delta_n3[i][0][j][0].reduce()
    return delta_n0, delta_n1, delta_n2, delta_n3


def _init_distance_dict(main_chain_len):
    delta_n0, delta_n1, delta_n2, delta_n3 = dict(), dict(), dict(), dict()
    for i in range(1, main_chain_len):
        delta_n0[i] = dict()
        delta_n1[i] = dict()
        delta_n2[i] = dict()
        delta_n3[i] = dict()
        delta_n0[i][0], delta_n0[i][1] = dict(), dict()
        delta_n1[i][0], delta_n1[i][1] = dict(), dict()
        delta_n2[i][0], delta_n2[i][1] = dict(), dict()
        delta_n3[i][0], delta_n3[i][1] = dict(), dict()
        for j in range(i + 1, main_chain_len + 1):
            delta_n0[i][0][j], delta_n0[i][1][j] = dict(), dict()
            delta_n1[i][0][j], delta_n1[i][1][j] = dict(), dict()
            delta_n2[i][0][j], delta_n2[i][1][j] = dict(), dict()
            delta_n3[i][0][j], delta_n3[i][1][j] = dict(), dict()

    return delta_n0, delta_n1, delta_n2, delta_n3


def _add_distances_side_chain(peptide, delta_n0, delta_n1, delta_n2,
                              delta_n3):
    """
    Calculates distances between beads located on side chains and adds the contribution to the
    distance calculated between beads (i and j) on the main chain. In the absence
    of side chains, this function returns a value of 0.

    Args:
        main_chain_len: Number of total beads in peptide
        delta_n0: Number of occurrences of axis 0 between beads
        delta_n1: Number of occurrences of axis 1 between beads
        delta_n2: Number of occurrences of axis 2 between beads
        delta_n3: Number of occurrences of axis 3 between beads

    Returns:
        delta_n0, delta_n1, delta_n2, delta_n3: Updated tuple (with added side chain
                                                contributions) that track the number
                                                of occurrences of turns at axes 0,1,2,3.
    """
    # TODO refactor try clauses
    main_chain_len = len(peptide.get_main_chain)
    for i in range(1, main_chain_len):  # j>i
        for j in range(i + 1, main_chain_len + 1):

            try:
                # TODO generalize to side chains longer than 1
                indic_0, indic_1, indic_2, indic_3 = peptide.get_main_chain[j].side_chain[0].get_indicator_functions()
                delta_n0[i][0][j][1] = (delta_n0[i][0][j][0] + (-1) ** j * indic_0[j][1]).reduce()
                delta_n1[i][0][j][1] = (delta_n1[i][0][j][0] + (-1) ** j * indic_1[j][1]).reduce()
                delta_n2[i][0][j][1] = (delta_n2[i][0][j][0] + (-1) ** j * indic_2[j][1]).reduce()
                delta_n3[i][0][j][1] = (delta_n3[i][0][j][0] + (-1) ** j * indic_3[j][1]).reduce()
            except:
                pass

            try:
                # TODO generalize to side chains longer than 1
                indic_0, indic_1, indic_2, indic_3 = peptide.get_main_chain[i].side_chain[0].get_indicator_functions()
                delta_n0[i][1][j][0] = (delta_n0[i][0][j][0] - (-1) ** i * indic_0[i][1]).reduce()
                delta_n1[i][1][j][0] = (delta_n1[i][0][j][0] - (-1) ** i * indic_1[i][1]).reduce()
                delta_n2[i][1][j][0] = (delta_n2[i][0][j][0] - (-1) ** i * indic_2[i][1]).reduce()
                delta_n3[i][1][j][0] = (delta_n3[i][0][j][0] - (-1) ** i * indic_3[i][1]).reduce()
            except:
                pass
            try:
                # TODO generalize to side chains longer than 1
                indic_0, indic_1, indic_2, indic_3 = peptide.get_main_chain[
                    j].side_chain[0].get_indicator_functions()
                # TODO generalize to side chains longer than 1
                indic_0, indic_1, indic_2, indic_3 = peptide.get_main_chain[
                    i].side_chain[0].get_indicator_functions()

                delta_n0[i][1][j][1] = (delta_n0[i][0][j][0] + (-1) ** j * indic_0[j][
                    1] - (-1) ** i * indic_0[i][1]).reduce()
                delta_n1[i][1][j][1] = (delta_n1[i][0][j][0] + (-1) ** j * indic_1[j][
                    1] - (-1) ** i * indic_1[i][1]).reduce()
                delta_n2[i][1][j][1] = (delta_n2[i][0][j][0] + (-1) ** j * indic_2[j][
                    1] - (-1) ** i * indic_2[i][1]).reduce()
                delta_n3[i][1][j][1] = (delta_n3[i][0][j][0] + (-1) ** j * indic_3[j][
                    1] - (-1) ** i * indic_3[i][1]).reduce()
            except:
                pass
    return delta_n0, delta_n1, delta_n2, delta_n3
