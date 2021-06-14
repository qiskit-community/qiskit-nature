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
import collections
from typing import List, Union

from qiskit.opflow import PauliSumOp, PauliOp

from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_full_identity
from problems.sampling.protein_folding.qubit_fixing import _fix_qubits
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide


# TODO refactor the data structure storing distances
def _calc_total_distances(peptide: Peptide):
    """
    Creates total distances between all bead pairs by summing the
    distances over all turns with axes, a = 0,1,2,3. For bead i with
    side chain s and bead j with side chain p, where j > i, the distance
    can be referenced as x_dist[i][p][j][s]

    Args:
        peptide: A Peptide object that includes all information about a protein.

    Returns:
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3.
    """
    delta_n0, delta_n1, delta_n2, delta_n3 = _calc_distances_main_chain(peptide)
    delta_n0, delta_n1, delta_n2, delta_n3 = _add_distances_side_chain(peptide, delta_n0, delta_n1,
                                                                       delta_n2, delta_n3)
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
                        x_dist[i][p][j][s] = _fix_qubits((delta_n0[i][p][j][s] ** 2 +
                                                          delta_n1[i][p][j][s] ** 2 +
                                                          delta_n2[i][p][j][s] ** 2 +
                                                          delta_n3[i][p][j][s] ** 2)).reduce()
                        r += 1
                    except KeyError:
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
        peptide: A Peptide object that includes all information about a protein.
    Returns:
        delta_n0, delta_n1, delta_n2, delta_n3: Tuple corresponding to
                                                the number of occurrences
                                                of turns at axes 0,1,2,3.
    """
    main_chain_len = len(peptide.get_main_chain)
    delta_n0, delta_n1, delta_n2, delta_n3 = _init_distance_dict(), _init_distance_dict(), \
                                             _init_distance_dict(), _init_distance_dict()
    for i in range(1, main_chain_len):
        for j in range(i + 1, main_chain_len + 1):
            delta_n0[i][0][j][0] = 0
            delta_n1[i][0][j][0] = 0
            delta_n2[i][0][j][0] = 0
            delta_n3[i][0][j][0] = 0
            for k in range(i, j):
                indic_0, indic_1, indic_2, indic_3 = peptide.get_main_chain[
                    k - 1].get_indicator_functions()
                delta_n0[i][0][j][0] += (-1) ** k * indic_0
                delta_n1[i][0][j][0] += (-1) ** k * indic_1
                delta_n2[i][0][j][0] += (-1) ** k * indic_2
                delta_n3[i][0][j][0] += (-1) ** k * indic_3
            delta_n0[i][0][j][0] = _fix_qubits(delta_n0[i][0][j][0]).reduce()
            delta_n1[i][0][j][0] = _fix_qubits(delta_n1[i][0][j][0]).reduce()
            delta_n2[i][0][j][0] = _fix_qubits(delta_n2[i][0][j][0]).reduce()
            delta_n3[i][0][j][0] = _fix_qubits(delta_n3[i][0][j][0]).reduce()

    return delta_n0, delta_n1, delta_n2, delta_n3


def _init_distance_dict():
    return collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))


def _add_distances_side_chain(peptide: Peptide, delta_n0, delta_n1, delta_n2,
                              delta_n3):
    """
    Calculates distances between beads located on side chains and adds the contribution to the
    distance calculated between beads (i and j) on the main chain. In the absence
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
    for i in range(1, main_chain_len):  # j>i
        for j in range(i + 1, main_chain_len + 1):

            if side_chain[j - 1]:
                try:
                    # TODO generalize to side chains longer than 1
                    indic_0, indic_1, indic_2, indic_3 = peptide.get_main_chain[j - 1].side_chain[
                        0].get_indicator_functions()
                    delta_n0[i][0][j][1] = _fix_qubits(
                        (delta_n0[i][0][j][0] + (-1) ** j * indic_0)).reduce()
                    delta_n1[i][0][j][1] = _fix_qubits(
                        (delta_n1[i][0][j][0] + (-1) ** j * indic_1)).reduce()
                    delta_n2[i][0][j][1] = _fix_qubits(
                        (delta_n2[i][0][j][0] + (-1) ** j * indic_2)).reduce()
                    delta_n3[i][0][j][1] = _fix_qubits(
                        (delta_n3[i][0][j][0] + (-1) ** j * indic_3)).reduce()
                except KeyError:
                    pass

            if side_chain[i - 1]:
                try:
                    # TODO generalize to side chains longer than 1
                    indic_0, indic_1, indic_2, indic_3 = peptide.get_main_chain[i - 1].side_chain[
                        0].get_indicator_functions()
                    delta_n0[i][1][j][0] = _fix_qubits(
                        (delta_n0[i][0][j][0] - (-1) ** i * indic_0)).reduce()
                    delta_n1[i][1][j][0] = _fix_qubits(
                        (delta_n1[i][0][j][0] - (-1) ** i * indic_1)).reduce()
                    delta_n2[i][1][j][0] = _fix_qubits(
                        (delta_n2[i][0][j][0] - (-1) ** i * indic_2)).reduce()
                    delta_n3[i][1][j][0] = _fix_qubits(
                        (delta_n3[i][0][j][0] - (-1) ** i * indic_3)).reduce()
                except KeyError:
                    pass

            if side_chain[i - 1] and side_chain[j - 1]:
                try:
                    # TODO generalize to side chains longer than 1
                    higher_indic_0, higher_indic_1, higher_indic_2, higher_indic_3 = \
                        peptide.get_main_chain[
                            j - 1].side_chain[0].get_indicator_functions()
                    # TODO generalize to side chains longer than 1
                    lower_indic_0, lower_indic_1, lower_indic_2, lower_indic_3 = \
                        peptide.get_main_chain[
                            i - 1].side_chain[0].get_indicator_functions()

                    delta_n0[i][1][j][1] = _fix_qubits(
                        (delta_n0[i][0][j][0] + (-1) ** j * higher_indic_0 - (
                            -1) ** i * lower_indic_0)).reduce()
                    delta_n1[i][1][j][1] = _fix_qubits(
                        (delta_n1[i][0][j][0] + (-1) ** j * higher_indic_1 - (
                            -1) ** i * lower_indic_1)).reduce()
                    delta_n2[i][1][j][1] = _fix_qubits(
                        (delta_n2[i][0][j][0] + (-1) ** j * higher_indic_2 - (
                            -1) ** i * lower_indic_2)).reduce()
                    delta_n3[i][1][j][1] = _fix_qubits(
                        (delta_n3[i][0][j][0] + (-1) ** j * higher_indic_3 - (
                            -1) ** i * lower_indic_3)).reduce()
                except KeyError:
                    pass
    return delta_n0, delta_n1, delta_n2, delta_n3


def _first_neighbor(i: int, p: int, j: int, s: int,
                    lambda_1: float, pair_energies: List[List[List[List[float]]]],
                    x_dist, pair_energies_multiplier: float = 0.1) -> Union[PauliSumOp, PauliOp]:
    """
    Creates first nearest neighbor interaction if beads are in contact
    and at a distance of 1 unit from each other. Otherwise, a large positive
    energetic penalty is added. Here, the penalty depends on the neighboring
    beads of interest (i and j), that is, lambda_0 > 6*(j -i + 1)*lambda_1 + e_ij.
    Here, we chose, lambda_0 = 7*(j- 1 + 1).

    Args:
        i: Backbone bead at turn i.
        j: Backbone bead at turn j (j > i).
        p: Side chain on backbone bead j.
        s: Side chain on backbone bead i.
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact.
        pair_energies: Numpy array of pair energies for amino acids.
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3.
        pair_energies_multiplier: A constant that multiplies pair energy contributions.

    Returns:
        expr: Contribution to energetic Hamiltonian.
    """
    bounding_constant = 7
    lambda_0 = bounding_constant * (j - i + 1) * lambda_1
    e = pair_energies[i, p, j, s]
    x = x_dist[i][p][j][s]
    expr = lambda_0 * (x - _build_full_identity(x.num_qubits))
    # + pair_energies_multiplier*e*_build_full_identity(x.num_qubits)
    return _fix_qubits(expr).reduce()


def _second_neighbor(i: int, p: int, j: int, s: int,
                     lambda_1: float, pair_energies: List[List[List[List[float]]]],
                     x_dist, pair_energies_multiplier: float = 0.1) -> Union[PauliSumOp, PauliOp]:
    """
    Creates energetic interaction that penalizes local overlap between
    beads that correspond to a nearest neighbor contact or adds no net
    interaction (zero) if beads are at a distance of 2 units from each other.
    Ensure second NN does not overlap with reference point

    Args:
        i: Backbone bead at turn i.
        j: Backbone bead at turn j (j > i).
        p: Side chain on backbone bead j.
        s: Side chain on backbone bead i.
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact.
        pair_energies: Numpy array of pair energies for amino acids.
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3.
        pair_energies_multiplier: A constant that multiplies pair energy contributions.

    Returns:
        expr: Contribution to energetic Hamiltonian.
    """
    e = pair_energies[i, p, j, s]
    x = x_dist[i][p][j][s]
    expr = lambda_1 * (2 * (_build_full_identity(
        x.num_qubits)) - x)  # + pair_energies_multiplier * e * _build_full_identity(x.num_qubits)
    return _fix_qubits(expr).reduce()
