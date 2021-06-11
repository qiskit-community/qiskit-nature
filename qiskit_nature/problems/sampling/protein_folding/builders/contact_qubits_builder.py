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

from qiskit.opflow import PauliSumOp, OperatorBase

from problems.sampling.protein_folding.peptide.pauli_ops_builder import _build_pauli_z_op, \
    _build_full_identity
from problems.sampling.protein_folding.peptide.peptide import Peptide


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
       pauli_contacts[lower_bead_id][p][upper_bead_id][s]
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
    for lower_bead_id in range(1, main_chain_len - 3):  # first qubit is number 1
        for upper_bead_id in range(lower_bead_id + 3, main_chain_len + 1):
            if (upper_bead_id - lower_bead_id) % 2:
                if (upper_bead_id - lower_bead_id) >= 5:
                    lower_main_upper_main[lower_bead_id][upper_bead_id] = _convert_to_qubits(
                        main_chain_len, (
                                full_id ^ _build_pauli_z_op(num_qubits, [lower_bead_id - 1,
                                                                         upper_bead_id - 1])))
                    print('possible contact between', lower_bead_id, '0 and', upper_bead_id, '0')
                    r_contact += 1
                if side_chain[lower_bead_id - 1] and side_chain[upper_bead_id - 1]:
                    lower_side_upper_side[lower_bead_id][upper_bead_id] = _convert_to_qubits(
                        main_chain_len, (
                                _build_pauli_z_op(num_qubits, [lower_bead_id - 1,
                                                               upper_bead_id - 1]) ^ full_id))
                    print('possible contact between', lower_bead_id, '1 and', upper_bead_id, '1')
                    r_contact += 1
            else:
                if (upper_bead_id - lower_bead_id) >= 4:
                    if side_chain[upper_bead_id - 1]:
                        print('possible contact between', lower_bead_id, '0 and', upper_bead_id,
                              '1')
                        main_op = full_id ^ _build_pauli_z_op(num_qubits, [lower_bead_id - 1])
                        side_op = _build_pauli_z_op(num_qubits, [upper_bead_id - 1]) ^ full_id
                        lower_side_upper_main[lower_bead_id][upper_bead_id] = _convert_to_qubits(
                            main_chain_len, main_op @ side_op)
                        r_contact += 1

                    if side_chain[lower_bead_id - 1]:
                        print('possible contact between', lower_bead_id, '1 and', upper_bead_id,
                              '0')
                        main_op = full_id ^ _build_pauli_z_op(num_qubits, [upper_bead_id - 1])
                        side_op = _build_pauli_z_op(num_qubits, [lower_bead_id - 1]) ^ full_id
                        lower_main_upper_side[lower_bead_id][upper_bead_id] = _convert_to_qubits(
                            main_chain_len, (side_op @ main_op))
                        r_contact += 1
    print('number of qubits required for contact : ', r_contact)
    return lower_main_upper_main, lower_side_upper_main, lower_main_upper_side, \
           lower_side_upper_side, r_contact


def _convert_to_qubits(main_chain_len: int, pauli_sum_op: PauliSumOp) -> OperatorBase:
    num_qubits_num = 2 * (main_chain_len - 1)
    full_id = _build_full_identity(num_qubits_num)
    return ((full_id ^ full_id) - pauli_sum_op) / 2



