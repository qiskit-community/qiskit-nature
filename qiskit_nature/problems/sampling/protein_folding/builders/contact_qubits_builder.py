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
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, PauliTable

from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_pauli_z_op, \
    _build_full_identity
from problems.sampling.protein_folding.peptide.peptide import Peptide


# TODO refactor data structures and try clauses
from problems.sampling.protein_folding.qubit_fixing import _fix_qubits


def _create_pauli_for_contacts(peptide: Peptide):
    """
    Creates Pauli operators for 1st nearest neighbor interactions

    Args:
        main_chain_len: Number of total beads in peptide
        side_chain: List of side chains in peptide

    Returns:
        pauli_contacts, r_contacts: Tuple consisting of dictionary
                                    of Pauli operators for contacts/
                                    interactions and number of qubits/
                                    contacts
       pauli_contacts[i][p][j][s]
    """
    main_chain_len = len(peptide.get_main_chain)
    side_chain = peptide.get_side_chain_hot_vector()
    pauli_contacts = _init_contacts_dict(main_chain_len)

    r_contact = 0
    num_qubits = 2 * (main_chain_len - 1)
    full_id = _build_full_identity(num_qubits)
    for i in range(1, main_chain_len - 3):  # first qubit is number 1
        for j in range(i + 3, main_chain_len + 1):
            if (j - i) % 2:
                if (j - i) >= 5:
                    pauli_contacts[i][0][j][0] = (
                            full_id ^ _build_pauli_z_op(num_qubits, [i - 1, j - 1]))
                    print('possible contact between', i, '0 and', j, '0')
                    r_contact += 1
                if side_chain[i - 1] and side_chain[j - 1]:
                    pauli_contacts[i][1][j][1] = (
                            _build_pauli_z_op(num_qubits, [i - 1, j - 1]) ^ full_id)
                    print('possible contact between', i, '1 and', j, '1')
                    r_contact += 1
            else:
                if (j - i) >= 4:
                    if side_chain[j - 1]:
                        print('possible contact between', i, '0 and', j, '1')
                        b = full_id ^ _build_pauli_z_op(num_qubits, [i - 1])
                        d = _build_pauli_z_op(num_qubits, [j - 1]) ^ full_id
                        pauli_contacts[i][0][j][1] = (b @ d)
                        r_contact += 1

                    if side_chain[i - 1]:
                        print('possible contact between', i, '1 and', j, '0')
                        b = full_id ^ _build_pauli_z_op(num_qubits, [j - 1])
                        d = _build_pauli_z_op(num_qubits, [i - 1]) ^ full_id
                        pauli_contacts[i][1][j][0] = (d @ b)
                        r_contact += 1
    print('number of qubits required for contact : ', r_contact)
    return pauli_contacts, r_contact


def _create_contact_qubits(main_chain_len, pauli_contacts):
    """
    Creates contact qubits to track 1st nearest
    neighbor interactions

    Args:
        main_chain_len: Number of total beads in peptide
        pauli_contacts: Dictionary of Pauli operators to track
                        contacts/interactions between beads

    Returns:
        contacts: Dictionary of contact qubits in symbolic notation
    """
    contacts = _init_contacts_dict(main_chain_len)
    num_qubits_num = 2 * (main_chain_len - 1)
    full_id = _build_full_identity(num_qubits_num)
    for i in range(1, main_chain_len - 3):  # first qubits is number 1
        for j in range(i + 4, main_chain_len + 1):
            for p in range(2):
                for s in range(2):
                    try:
                        contacts[i][p][j][s] = ((full_id ^ full_id) - pauli_contacts[i][p][j][
                            s]) / 2
                    except KeyError:
                        pass
    return contacts


def _init_contacts_dict(main_chain_len):
    pauli_contacts = dict()
    for i in range(1, main_chain_len - 3):
        pauli_contacts[i] = dict()
        pauli_contacts[i][0] = dict()
        pauli_contacts[i][1] = dict()
        for j in range(i + 3, main_chain_len + 1):
            pauli_contacts[i][0][j] = dict()
            pauli_contacts[i][1][j] = dict()
    return pauli_contacts


# gathers qubits from conformation and qubits from NN intraction
def _create_new_qubit_list(peptide: Peptide, pauli_contacts):
    """
    Creates new set of contact qubits for second nearest neighbour
    interactions. Note, the need of multiple interaction qubits
    for each i,j pair.

    Args:
        main_chain_len: Number of total beads in peptide
        side_chain: List of side chains in peptide
        pauli_conf: Dictionary of Pauli operators to track conformation
        pauli_contacts: Dictionary of Pauli operators to track contacts between beads

    Returns:
        new_qubits: Dictionary of qubits in symbolic notation
    """
    main_chain_len = len(peptide.get_main_chain)
    side_chain = peptide.get_side_chain_hot_vector()
    old_qubits_conf = []
    old_qubits_contact = []
    num_qubits = 2 * (main_chain_len - 1)
    full_id = _build_full_identity(num_qubits)
    for q in range(3, main_chain_len):
        if q != 3:
            old_qubits_conf.append(full_id ^ peptide.get_main_chain[q - 1].turn_qubits[0])
            old_qubits_conf.append(full_id ^ peptide.get_main_chain[q - 1].turn_qubits[1])
        else:
            old_qubits_conf.append(full_id ^ peptide.get_main_chain[q - 1].turn_qubits[0])
        if side_chain[q - 1]:
            old_qubits_conf.append(
                peptide.get_main_chain[q - 1].side_chain[0].turn_qubits[0] ^ full_id)
            old_qubits_conf.append(
                peptide.get_main_chain[q - 1].side_chain[0].turn_qubits[1] ^ full_id)

    for i in range(1, main_chain_len - 3):
        for j in range(i + 4, main_chain_len + 1):
            for p in range(2):
                for s in range(2):
                    try:
                        old_qubits_contact.append(
                            0.5 * ((full_id ^ full_id) - pauli_contacts[i][p][j][s]))
                    except KeyError:
                        pass

    new_qubits = [0] + old_qubits_conf + old_qubits_contact
    return new_qubits


def _first_neighbor(i: int, p: int, j: int, s: int,
                    lambda_1, pair_energies,
                    x_dist, pair_energies_multiplier: float = 0.1):
    """
    Creates first nearest neighbor interaction if beads are in contact
    and at a distance of 1 unit from each other. Otherwise, a large positive
    energetic penalty is added. Here, the penalty depends on the neighboring
    beads of interest (i and j), that is, lambda_0 > 6*(j -i + 1)*lambda_1 + e_ij.
    Here, we chose, lambda_0 = 7*(j- 1 + 1).

    Args:
        i: Backbone bead at turn i
        j: Backbone bead at turn j (j > i)
        p: Side chain on backbone bead j
        s: Side chain on backbone bead i
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact
        pair_energies: Numpy array of pair energies for amino acids
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3
        pauli_conf: Dictionary of conformation Pauli operators in symbolic notation

    Returns:
        expr: Contribution to energetic Hamiltonian in symbolic notation
    """
    lambda_0 = 7 * (j - i + 1) * lambda_1
    e = pair_energies[i, p, j, s]
    x = x_dist[i][p][j][s]
    expr = lambda_0 * (x - _build_full_identity(x.num_qubits))
    # + pair_energies_multiplier*e*_build_full_identity(x.num_qubits)
    return _fix_qubits(expr).reduce()


def _second_neighbor(i: int, p: int, j: int, s: int,
                     lambda_1, pair_energies,
                     x_dist, pair_energies_multiplier: float = 0.1):
    """
    Creates energetic interaction that penalizes local overlap between
    beads that correspond to a nearest neighbor contact or adds no net
    interaction (zero) if beads are at a distance of 2 units from each other.
    Ensure second NN does not overlap with reference point

    Args:
        i: Backbone bead at turn i
        j: Backbone bead at turn j (j > i)
        p: Side chain on backbone bead j
        s: Side chain on backbone bead i
        lambda_1: Constraint to penalize local overlap between
                 beads within a nearest neighbor contact
        pair_energies: Numpy array of pair energies for amino acids
        x_dist: Numpy array that tracks all distances between backbone and side chain
                beads for all axes: 0,1,2,3
        pauli_conf: Dictionary of conformation Pauli operators in symbolic notation

    Returns:
        expr: Contribution to energetic Hamiltonian in symbolic notation
    """
    e = pair_energies[i, p, j, s]
    x = x_dist[i][p][j][s]
    expr = lambda_1 * (2 * (_build_full_identity(
        x.num_qubits)) - x) #+ pair_energies_multiplier * e * _build_full_identity(x.num_qubits)
    return _fix_qubits(expr).reduce()
