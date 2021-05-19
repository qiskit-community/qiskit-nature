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
from typing import List

from qiskit.opflow import PauliOp, I, Z


def _build_full_identity(num_turn_qubits) -> PauliOp:
    FULL_ID = I
    for _ in range(1, num_turn_qubits):
        FULL_ID = I ^ FULL_ID
    return FULL_ID


def _build_pauli_z_op(num_qubits: int, pauli_z_indices: List[int]):
    if 0 in pauli_z_indices:
        temp = I
    else:
        temp = Z
    for i in range(1, num_qubits):
        if i in pauli_z_indices:
            temp = Z ^ temp
        else:
            temp = I ^ temp

    return temp
