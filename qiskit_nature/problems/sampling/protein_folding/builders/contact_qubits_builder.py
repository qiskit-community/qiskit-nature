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

from qiskit.opflow import PauliSumOp

from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_pauli_z_op, \
    _build_full_identity
from problems.sampling.protein_folding.peptide.peptide import Peptide

# TODO refactor data structures and try clauses
from problems.sampling.protein_folding.qubit_fixing import _fix_qubits


def _create_contact_qubits(peptide: Peptide):
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

    lower_main_upper_main = collections.defaultdict(dict)
    lower_side_upper_main = collections.defaultdict(dict)
    lower_main_upper_side = collections.defaultdict(dict)
    lower_side_upper_side = collections.defaultdict(dict)

    r_contact = 0
    num_qubits = 2 * (main_chain_len - 1)
    full_id = _build_full_identity(num_qubits)
    for i in range(1, main_chain_len - 3):  # first qubit is number 1
        for j in range(i + 3, main_chain_len + 1):
            if (j - i) % 2:
                if (j - i) >= 5:
                    lower_main_upper_main[i][j] = _convert_to_qubits(main_chain_len, (
                            full_id ^ _build_pauli_z_op(num_qubits, [i - 1, j - 1])))
                    print('possible contact between', i, '0 and', j, '0')
                    r_contact += 1
                if side_chain[i - 1] and side_chain[j - 1]:
                    lower_side_upper_side[i][j] = _convert_to_qubits(main_chain_len, (
                            _build_pauli_z_op(num_qubits, [i - 1, j - 1]) ^ full_id))
                    print('possible contact between', i, '1 and', j, '1')
                    r_contact += 1
            else:
                if (j - i) >= 4:
                    if side_chain[j - 1]:
                        print('possible contact between', i, '0 and', j, '1')
                        b = full_id ^ _build_pauli_z_op(num_qubits, [i - 1])
                        d = _build_pauli_z_op(num_qubits, [j - 1]) ^ full_id
                        lower_side_upper_main[i][j] = _convert_to_qubits(main_chain_len, b @ d)
                        r_contact += 1

                    if side_chain[i - 1]:
                        print('possible contact between', i, '1 and', j, '0')
                        b = full_id ^ _build_pauli_z_op(num_qubits, [j - 1])
                        d = _build_pauli_z_op(num_qubits, [i - 1]) ^ full_id
                        lower_main_upper_side[i][j] = _convert_to_qubits(main_chain_len, (d @ b))
                        r_contact += 1
    print('number of qubits required for contact : ', r_contact)
    return lower_main_upper_main, lower_side_upper_main, lower_main_upper_side, \
           lower_side_upper_side, r_contact


def _convert_to_qubits(main_chain_len: int, pauli_sum_op: PauliSumOp):
    num_qubits_num = 2 * (main_chain_len - 1)
    full_id = _build_full_identity(num_qubits_num)
    return ((full_id ^ full_id) - pauli_sum_op) / 2


# gathers qubits from conformation and qubits from NN intraction


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
        x.num_qubits)) - x)  # + pair_energies_multiplier * e * _build_full_identity(x.num_qubits)
    return _fix_qubits(expr).reduce()
