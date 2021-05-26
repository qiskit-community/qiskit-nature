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
import numpy as np
from qiskit.opflow import I, PauliSumOp, OperatorBase
from qiskit.quantum_info import SparsePauliOp, PauliTable

from problems.sampling.protein_folding.builders.contact_qubits_builder import _first_neighbor, \
    _second_neighbor
from qiskit_nature.problems.sampling.protein_folding.peptide.beads.base_bead import BaseBead
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide


def _build_qubit_op(n, side_chain, pair_energies, lambda_chiral, lambda_back, lambda_1,
                    lambda_contacts, n_contacts):
    pass


def _check_turns(lower_bead: BaseBead, higher_bead: BaseBead) -> OperatorBase:
    lower_bead_indic_0, lower_bead_indic_1, lower_bead_indic_2, lower_bead_indic_3 = \
        lower_bead.get_indicator_functions()

    higher_bead_indic_0, higher_bead_indic_1, higher_bead_indic_2, higher_bead_indic_3 = \
        higher_bead.get_indicator_functions()

    t_ij = lower_bead_indic_0 @ higher_bead_indic_0 + lower_bead_indic_1 @ higher_bead_indic_1 + \
           lower_bead_indic_2 @ higher_bead_indic_2 + lower_bead_indic_3 @ higher_bead_indic_3
    return t_ij


def _create_h_back(peptide: Peptide, lambda_back):
    main_chain = peptide.get_main_chain
    main_chain_len = len(main_chain)
    h_back = PauliSumOp.from_list(
        [("I" * 2 * (main_chain_len - 1), 0)])
    for i in range(len(main_chain) - 2):
        h_back += lambda_back * _check_turns(main_chain[i], main_chain[i + 1])

    h_back = _set_binaries(h_back).reduce()
    return h_back


def _set_binaries(H_back):
    new_tables = []
    new_coeffs = []
    for i in range(len(H_back)):
        H = H_back[i]
        table_Z = np.copy(H.primitive.table.Z[0])
        table_X = np.copy(H.primitive.table.X[0])
        # get coeffs and update
        coeffs = np.copy(H.primitive.coeffs[0])
        if table_Z[1] == np.bool_(True):
            coeffs = -1 * coeffs
        if table_Z[5] == np.bool_(True):
            coeffs = -1 * coeffs
        # impose preset binary values
        table_Z[0] = np.bool_(False)
        table_Z[1] = np.bool_(False)
        table_Z[2] = np.bool_(False)
        table_Z[3] = np.bool_(False)
        table_Z[5] = np.bool_(False)
        new_table = np.concatenate((table_X, table_Z), axis=0)
        new_tables.append(new_table)
        new_coeffs.append(coeffs)
    new_pauli_table = PauliTable(data=new_tables)
    H_back_updated = PauliSumOp(SparsePauliOp(data=new_pauli_table, coeffs=new_coeffs))
    H_back_updated = H_back_updated.reduce()
    return H_back_updated


def _create_H_chiral(peptide, lambda_chiral):
    """
    Creates a penalty/constrain term to the total Hamiltonian that imposes that all the position
    of all side chain beads impose the right chirality. Note that the position of the side chain
    bead at a location (i) is determined by the turn indicators at i - 1 and i. In the absence
    of side chains, this function returns a value of 0.

    Args:
        N: Number of total beads in peptide
        side_chain: List of side chains in peptide
        lambda_chiral: Penalty/constraint to impose the right chirality
        indic_0: Turn indicator for axis 0
        indic_1: Turn indicator for axis 1
        indic_2: Turn indicator for axis 2
        indic_3: Turn indicator for axis 3
        pauli_conf: Dictionary of conformation Pauli operators in symbolic notation

    Returns:
        H_chiral: Hamiltonian term in symbolic notation that imposes the right chirality
    """

    main_chain = peptide.get_main_chain
    main_chain_len = len(main_chain)
    H_chiral = PauliSumOp.from_list(
        [("I" * 2 * (main_chain_len - 1), 0)])  # TODO make sure correct size
    for i in range(1, len(main_chain)):  # TODO double check range
        higher_main_bead = main_chain[i]

        if not higher_main_bead.side_chain:
            continue

        higher_side_bead = higher_main_bead.side_chain[0]

        lower_main_bead = main_chain[i - 1]

        lower_main_bead_indic_0, lower_main_bead_indic_1, lower_main_bead_indic_2, \
        lower_main_bead_indic_3 = \
            lower_main_bead.get_indicator_functions()

        higher_main_bead_indic_0, higher_main_bead_indic_1, higher_main_bead_indic_2, \
        higher_main_bead_indic_3 = \
            higher_main_bead.get_indicator_functions()
        higher_side_bead_indic_0, higher_side_bead_indic_1, higher_side_bead_indic_2, \
        higher_side_bead_indic_3 = \
            higher_side_bead.get_indicator_functions()

        si = int((1 - (-1) ** i) / 2)
        H_chiral += lambda_chiral * (I - higher_side_bead_indic_0) @ ((1 - si) * (
                lower_main_bead_indic_1 @ higher_main_bead_indic_2 + lower_main_bead_indic_2 @
                higher_main_bead_indic_3 +
                lower_main_bead_indic_3 @ higher_main_bead_indic_1) + si * (
                                                                              lower_main_bead_indic_2 @ higher_main_bead_indic_1 +
                                                                              lower_main_bead_indic_3 @ higher_main_bead_indic_2 +
                                                                              lower_main_bead_indic_1 @ higher_main_bead_indic_3))
        H_chiral += lambda_chiral * (I - higher_side_bead_indic_1) @ ((1 - si) * (
                lower_main_bead_indic_0 @ higher_main_bead_indic_3 + lower_main_bead_indic_2 @
                higher_main_bead_indic_0 +
                lower_main_bead_indic_3 @ higher_main_bead_indic_2) + si * (
                                                                              lower_main_bead_indic_3 @ higher_main_bead_indic_0 +
                                                                              lower_main_bead_indic_0 @ higher_main_bead_indic_2 +
                                                                              lower_main_bead_indic_2 @ higher_main_bead_indic_3))
        H_chiral += lambda_chiral * (I - higher_side_bead_indic_2) @ ((1 - si) * (
                lower_main_bead_indic_0 @ higher_main_bead_indic_1 + lower_main_bead_indic_1 @
                higher_main_bead_indic_3 +
                lower_main_bead_indic_3 @ higher_main_bead_indic_0) + si * (
                                                                              lower_main_bead_indic_1 @ higher_main_bead_indic_0 +
                                                                              lower_main_bead_indic_3 @ higher_main_bead_indic_1 +
                                                                              lower_main_bead_indic_0 @ higher_main_bead_indic_3))
        H_chiral += lambda_chiral * (I - higher_side_bead_indic_3) @ ((1 - si) * (
                lower_main_bead_indic_0 @ higher_main_bead_indic_2 + lower_main_bead_indic_1 @
                higher_main_bead_indic_0 +
                lower_main_bead_indic_2 @ higher_main_bead_indic_1) + si * (
                                                                              lower_main_bead_indic_2 @ higher_main_bead_indic_0 +
                                                                              lower_main_bead_indic_0 @ higher_main_bead_indic_1 +
                                                                              lower_main_bead_indic_1 @ higher_main_bead_indic_2))
        H_chiral = _set_binaries(H_chiral).reduce()
    return H_chiral


def _create_H_BBBB(main_chain_len, lambda_1, pair_energies,
                   x_dist, contacts):
    """
    Creates Hamiltonian term corresponding to 1st neighbor interaction between
    main/backbone (BB) beads

    Args:
        main_chain_len: Number of total beads in peptide
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact
        pair_energies: Numpy array of pair energies for amino acids
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3
        pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
        contacts: Dictionary of contact qubits in symbolic notation

    Returns:
        H_BBBB: Hamiltonian term in symbolic notation
    """
    H_BBBB = PauliSumOp.from_list(
        [("I" * 2 * (main_chain_len - 1), 0)])  # TODO make sure correct size
    for i in range(1, main_chain_len - 3):
        for j in range(i + 5, main_chain_len + 1):
            if (j - i) % 2 == 0:
                continue
            else:
                H_BBBB += contacts[i][0][j][0] * _first_neighbor(i, 0, j, 0, lambda_1,
                                                                 pair_energies,
                                                                 x_dist)
                try:
                    H_BBBB += contacts[i][0][j][0] * _second_neighbor(i - 1, 0, j, 0, lambda_1,
                                                                      pair_energies, x_dist)
                except:
                    pass
                try:
                    H_BBBB += contacts[i][0][j][0] * _second_neighbor(i + 1, 0, j, 0, lambda_1,
                                                                      pair_energies, x_dist)
                except:
                    pass
                try:
                    H_BBBB += contacts[i][0][j][0] * _second_neighbor(i, 0, j - 1, 0, lambda_1,
                                                                      pair_energies, x_dist)
                except:
                    pass
                try:
                    H_BBBB += contacts[i][0][j][0] * _second_neighbor(i, 0, j + 1, 0, lambda_1,
                                                                      pair_energies, x_dist)
                except:
                    pass
            H_BBBB = _set_binaries(H_BBBB).reduce()
    return H_BBBB


def _create_H_BBSC_and_H_SCBB(main_chain_len, side_chain, lambda_1,
                              pair_energies, x_dist,
                              contacts):
    """
    Creates Hamiltonian term corresponding to 1st neighbor interaction between
    main/backbone (BB) and side chain (SC) beads. In the absence
    of side chains, this function returns a value of 0.

    Args:
        main_chain_len: Number of total beads in peptide
        side: List of side chains in peptide
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact
        pair_energies: Numpy array of pair energies for amino acids
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3
        pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
        contacts: Dictionary of contact qubits in symbolic notation

    Returns:
        H_BBSC, H_SCBB: Tuple of Hamiltonian terms consisting of backbone and side chain 
        interactions
    """
    H_BBSC = PauliSumOp.from_list(
        [("I" * 2 * (main_chain_len - 1), 0)])  # TODO make sure correct size
    H_SCBB = PauliSumOp.from_list(
        [("I" * 2 * (main_chain_len - 1), 0)])  # TODO make sure correct size
    for i in range(1, main_chain_len - 3):
        for j in range(i + 4, main_chain_len + 1):
            if (j - i) % 2 == 1:
                continue
            else:
                if side_chain[j - 1] == 1:
                    H_BBSC += contacts[i][1][j][0] * (
                            _first_neighbor(i, 1, j, 0, lambda_1, pair_energies, x_dist) +
                            _second_neighbor(i, 0, j, 0, lambda_1, pair_energies, x_dist))
                    try:
                        H_BBSC += contacts[i][1][j][0] * _first_neighbor(i, 1, j, 1, lambda_1,
                                                                         pair_energies, x_dist)
                    except:
                        pass
                    try:
                        H_BBSC += contacts[i][1][j][0] * _second_neighbor(i + 1, 1, j, 0, lambda_1,
                                                                          pair_energies, x_dist)
                    except:
                        pass
                    try:
                        H_BBSC += contacts[i][1][j][0] * _second_neighbor(i - 1, 1, j, 0, lambda_1,
                                                                          pair_energies, x_dist)
                    except:
                        pass
                    H_BBSC = H_BBSC.reduce()
                if side_chain[i - 1] == 1:
                    H_SCBB += contacts[i][0][j][1] * (
                            _first_neighbor(i, 0, j, 1, lambda_1, pair_energies, x_dist) +
                            _second_neighbor(i, 0, j, 0, lambda_1, pair_energies, x_dist))
                    try:
                        H_SCBB += contacts[i][0][j][1] * _second_neighbor(i, 0, j, 1, lambda_1,
                                                                          pair_energies, x_dist)
                    except:
                        pass
                    try:
                        H_SCBB += contacts[i][0][j][1] * _second_neighbor(i, 0, j + 1, 1, lambda_1,
                                                                          pair_energies, x_dist)
                    except:
                        pass
                    try:
                        H_SCBB += contacts[i][0][j][1] * _second_neighbor(i, 0, j - 1, 1, lambda_1,
                                                                          pair_energies, x_dist)
                    except:
                        pass
                    H_SCBB = H_SCBB.reduce()
    return _set_binaries(H_BBSC), _set_binaries(H_SCBB)


def _create_H_SCSC(main_chain_len, side_chain, lambda_1,
                   pair_energies, x_dist, contacts):
    """
    Creates Hamiltonian term corresponding to 1st neighbor interaction between
    side chain (SC) beads. In the absence of side chains, this function
    returns a value of 0.

    Args:
        main_chain_len: Number of total beads in peptides
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact
        pair_energies: Numpy array of pair energies for amino acids
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3
        pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
        contacts: Dictionary of contact qubits in symbolic notation

    Returns:
        H_SCSC: Hamiltonian term consisting of side chain pairwise interactions
    """
    H_SCSC = PauliSumOp.from_list(
        [("I" * 2 * (main_chain_len - 1), 0)])  # TODO make sure correct size
    for i in range(1, main_chain_len - 3):
        for j in range(i + 5, main_chain_len + 1):
            if (j - i) % 2 == 0:
                continue
            if side_chain[i - 1] == 0 or side_chain[j - 1] == 0:
                continue
            H_SCSC += contacts[i][1][j][1] * (
                    _first_neighbor(i, 1, j, 1, lambda_1, pair_energies, x_dist) +
                    _second_neighbor(i, 1, j, 0, lambda_1, pair_energies, x_dist)
                    +
                    _second_neighbor(i, 0, j, 1, lambda_1, pair_energies, x_dist))
            H_SCSC = H_SCSC.reduce()
    return _set_binaries(H_SCSC)


def _create_H_short(main_chain_len, side_chain, pair_energies,
                    x_dist, indic_0,
                    indic_1, indic_2, indic_3):
    """
    Creates Hamiltonian constituting interactions between beads that are no more than
    4 beads apart. If no side chains are present, this function returns 0.

    Args:
        main_chain_len: Number of total beads in peptide
        side_chain: List of side chains in peptide
        pair_energies: Numpy array of pair energies for amino acids
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3
        pauli_conf: Dictionary of conformation Pauli operators in symbolic notation
        indic_0: Turn indicator for axis 0
        indic_1: Turn indicator for axis 1
        indic_2: Turn indicator for axis 2
        indic_3: Turn indicator for axis 3

    Returns:
        H_short: Contribution to energetic Hamiltonian in symbolic notation t
    """
    H_short = PauliSumOp.from_list(
        [("I" * 2 * (main_chain_len - 1), 0)])  # TODO make sure correct size
    for i in range(1, main_chain_len - 2):
        # checks interactions between beads no more than 4 beads apart
        if side_chain[i - 1] == 1 and side_chain[i + 2] == 1:
            H_short += _simplify(pauli_conf,
                                 _check_turns(i, 1, i + 2, 0, indic_0, indic_1, indic_2, indic_3,
                                              pauli_conf) * \
                                 _check_turns(i + 3, 1, i, 0, indic_0, indic_1, indic_2, indic_3,
                                              pauli_conf)) * \
                       (pair_energies[i, 1, i + 3, 1] + 0.1 * (
                               pair_energies[i, 1, i + 3, 0] + pair_energies[i, 0, i + 3, 1]))
    return _set_binaries(H_short)
