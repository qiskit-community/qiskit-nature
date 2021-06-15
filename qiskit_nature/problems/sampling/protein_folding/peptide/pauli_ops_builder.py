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
"""Builds Pauli operators of a given size."""
from typing import List

from qiskit.opflow import PauliOp, I, Z


def _build_full_identity(num_qubits: int) -> PauliOp:
    """
    Builds a full identity operator of a given size.
    Args:
        num_qubits: number of qubits on which a full identity operator will be created.

    Returns:
        full_identity: a full identity operator of a given size.
    """
    full_identity = I
    for _ in range(1, num_qubits):
        full_identity = I ^ full_identity
    return full_identity


def _build_pauli_z_op(num_qubits: int, pauli_z_indices: List[int]) -> PauliOp:
    """
    Builds a Pauli operator of a given size with Pauli Z operators on indicated positions and
    identity operators on other positions.
    Args:
        num_qubits: number of qubits on which a Pauli operator will be created.
        pauli_z_indices: a list of indices in a Pauli operator on which a Pauli Z operator shall
                        appear.

    Returns:
        operator: a Pauli operator of a given size with Pauli Z operators on indicated positions
                andidentity operators on other positions.
    """
    if 0 in pauli_z_indices:
        operator = Z
    else:
        operator = I
    for i in range(1, num_qubits):
        if i in pauli_z_indices:
            operator = Z ^ operator
        else:
            operator = I ^ operator

    return operator
