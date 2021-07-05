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
"""Builds a contact map that stores contacts between beads in a peptide."""
import collections
import logging
from typing import Tuple, Dict

from qiskit.opflow import PauliSumOp, OperatorBase

from ..peptide.pauli_ops_builder import (
    _build_pauli_z_op,
    _build_full_identity,
)
from ..peptide.peptide import Peptide

logger = logging.getLogger(__name__)


def _create_contact_qubits(
    peptide: Peptide,
) -> Tuple[Dict[int, dict], Dict[int, dict], Dict[int, dict], Dict[int, dict], int]:
    """
    Creates Pauli operators for 3rd+ nearest neighbor interactions. The type of operator depends
    on whether beads belong to the same set (see also https://arxiv.org/pdf/1908.02163.pdf), how
    far they are from each other and whether they host a side chain.

    Args:
        peptide: A Peptide object that includes all information about a protein.

    Returns:
        lower_main_upper_main, lower_side_upper_main, lower_main_upper_side, \
           lower_side_upper_side, num_contacts: Tuple consisting of dictionaries
                                    of Pauli operators for contacts/
                                    interactions between a lower/upper bead from the main/side
                                    chain and a lower/upper bead from the main/side chain and the
                                    total number of contacts.
    """
    main_chain_len = len(peptide.get_main_chain)
    side_chain = peptide.get_side_chain_hot_vector()

    lower_main_upper_main: Dict[int, Dict[int, OperatorBase]] = collections.defaultdict(dict)
    lower_side_upper_main: Dict[int, Dict[int, OperatorBase]] = collections.defaultdict(dict)
    lower_main_upper_side: Dict[int, Dict[int, OperatorBase]] = collections.defaultdict(dict)
    lower_side_upper_side: Dict[int, Dict[int, OperatorBase]] = collections.defaultdict(dict)

    num_contacts = 0
    num_qubits = 2 * (main_chain_len - 1)
    full_id = _build_full_identity(num_qubits)
    for lower_bead_id in range(1, main_chain_len - 3):  # first qubit is number 1
        for upper_bead_id in range(
            lower_bead_id + 3, main_chain_len + 1
        ):  # interactions between beads that are nearest or second nearest neighbor do not help
            # discriminating the folds, see https://arxiv.org/pdf/1908.02163.pdf section C
            if _are_beads_in_different_sets(upper_bead_id, lower_bead_id):
                if _are_beads_k_plus_steps_apart(upper_bead_id, lower_bead_id, k=5):
                    lower_main_upper_main[lower_bead_id][upper_bead_id] = _convert_to_qubits(
                        main_chain_len,
                        (
                            full_id
                            ^ _build_pauli_z_op(num_qubits, [lower_bead_id - 1, upper_bead_id - 1])
                        ),
                    )
                    _log_contact(lower_bead_id, upper_bead_id, "main_chain", "main_chain")
                    num_contacts += 1
                if side_chain[lower_bead_id - 1] and side_chain[upper_bead_id - 1]:
                    lower_side_upper_side[lower_bead_id][upper_bead_id] = _convert_to_qubits(
                        main_chain_len,
                        (
                            _build_pauli_z_op(num_qubits, [lower_bead_id - 1, upper_bead_id - 1])
                            ^ full_id
                        ),
                    )
                    _log_contact(lower_bead_id, upper_bead_id, "side_chain", "side_chain")
                    num_contacts += 1
            else:
                if _are_beads_k_plus_steps_apart(upper_bead_id, lower_bead_id, k=4):
                    if side_chain[upper_bead_id - 1]:
                        _log_contact(lower_bead_id, upper_bead_id, "main_chain", "side_chain")
                        main_op = full_id ^ _build_pauli_z_op(num_qubits, [lower_bead_id - 1])
                        side_op = _build_pauli_z_op(num_qubits, [upper_bead_id - 1]) ^ full_id
                        lower_side_upper_main[lower_bead_id][upper_bead_id] = _convert_to_qubits(
                            main_chain_len, main_op @ side_op
                        )
                        num_contacts += 1

                    if side_chain[lower_bead_id - 1]:
                        _log_contact(lower_bead_id, upper_bead_id, "side_chain", "main_chain")
                        main_op = full_id ^ _build_pauli_z_op(num_qubits, [upper_bead_id - 1])
                        side_op = _build_pauli_z_op(num_qubits, [lower_bead_id - 1]) ^ full_id
                        lower_main_upper_side[lower_bead_id][upper_bead_id] = _convert_to_qubits(
                            main_chain_len, (side_op @ main_op)
                        )
                        num_contacts += 1
    logger.info("number of qubits required for contact %s:", num_contacts)
    return (
        lower_main_upper_main,
        lower_side_upper_main,
        lower_main_upper_side,
        lower_side_upper_side,
        num_contacts,
    )


# the paper (https://arxiv.org/pdf/1908.02163.pdf) defines sets A and B; beads' membership
# alternate between A and B
def _are_beads_in_different_sets(upper_bead_id: int, lower_bead_id: int) -> bool:
    return (upper_bead_id - lower_bead_id) % 2 == 1


def _are_beads_k_plus_steps_apart(upper_bead_id: int, lower_bead_id: int, k: int) -> bool:
    return (upper_bead_id - lower_bead_id) >= k


def _log_contact(lower_bead_id, upper_bead_id, lower_chain_type, upper_chain_type):
    logger.info(
        "possible contact between a bead %s on the %s and a bead %s on the %s",
        lower_bead_id,
        lower_chain_type,
        upper_bead_id,
        upper_chain_type,
    )


def _convert_to_qubits(main_chain_len: int, pauli_sum_op: PauliSumOp) -> OperatorBase:
    num_qubits_num = 2 * (main_chain_len - 1)
    full_id = _build_full_identity(num_qubits_num)
    return ((full_id ^ full_id) - pauli_sum_op) / 2
