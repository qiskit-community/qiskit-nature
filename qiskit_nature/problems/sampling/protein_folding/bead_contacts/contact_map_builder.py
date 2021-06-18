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
from typing import Tuple, DefaultDict, Dict

from qiskit.opflow import PauliSumOp, OperatorBase

from ..peptide.pauli_ops_builder import (
    _build_pauli_z_op,
    _build_full_identity,
)
from ..peptide.peptide import Peptide

logger = logging.getLogger(__name__)


def _create_contact_qubits(
    peptide: Peptide,
) -> Tuple[
    DefaultDict[int, dict],
    DefaultDict[int, dict],
    DefaultDict[int, dict],
    DefaultDict[int, dict],
    int,
]:
    """
    Creates Pauli operators for 1st nearest neighbor interactions

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

    lower_main_upper_main: DefaultDict[int, Dict[int, OperatorBase]] = collections.defaultdict(dict)
    lower_side_upper_main: DefaultDict[int, Dict[int, OperatorBase]] = collections.defaultdict(dict)
    lower_main_upper_side: DefaultDict[int, Dict[int, OperatorBase]] = collections.defaultdict(dict)
    lower_side_upper_side: DefaultDict[int, Dict[int, OperatorBase]] = collections.defaultdict(dict)

    num_contacts = 0
    num_qubits = 2 * (main_chain_len - 1)
    full_id = _build_full_identity(num_qubits)
    for lower_bead_id in range(1, main_chain_len - 3):  # first qubit is number 1
        for upper_bead_id in range(lower_bead_id + 3, main_chain_len + 1):
            if (upper_bead_id - lower_bead_id) % 2:
                if (upper_bead_id - lower_bead_id) >= 5:
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
                if (upper_bead_id - lower_bead_id) >= 4:
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
    logger.info(f"number of qubits required for contact : {num_contacts}")
    return (
        lower_main_upper_main,
        lower_side_upper_main,
        lower_main_upper_side,
        lower_side_upper_side,
        num_contacts,
    )


def _log_contact(lower_bead_id, upper_bead_id, lower_chain_type, upper_chain_type):
    logger.info(
        f"possible contact between a bead {lower_bead_id} on the {lower_chain_type} and a bead "
        f"{upper_bead_id} on the {upper_chain_type}"
    )


def _convert_to_qubits(main_chain_len: int, pauli_sum_op: PauliSumOp) -> OperatorBase:
    num_qubits_num = 2 * (main_chain_len - 1)
    full_id = _build_full_identity(num_qubits_num)
    return ((full_id ^ full_id) - pauli_sum_op) / 2
