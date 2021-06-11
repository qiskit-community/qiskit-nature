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
from typing import Union

import numpy as np
from qiskit.opflow import PauliSumOp, OperatorBase, PauliOp
from qiskit.quantum_info import PauliTable, SparsePauliOp, Pauli


def _fix_qubits(qubits: Union[PauliSumOp, PauliOp, OperatorBase]):
    new_tables = []
    new_coeffs = []
    if isinstance(qubits, PauliOp):
        table_z = np.copy(qubits.primitive.z)
        table_x = np.copy(qubits.primitive.x)
        _preset_binary_vals(table_z)
        return PauliOp(Pauli((table_z, table_x)))

    for i in range(len(qubits)):
        h = qubits[i]
        table_z = np.copy(h.primitive.table.Z[0])
        table_x = np.copy(h.primitive.table.X[0])
        coeffs = _calc_updated_coeffs(h, table_z)
        _preset_binary_vals(table_z)
        new_table = np.concatenate((table_x, table_z), axis=0)
        new_tables.append(new_table)
        new_coeffs.append(coeffs)
    new_pauli_table = PauliTable(data=new_tables)
    qubits_updated = PauliSumOp(SparsePauliOp(data=new_pauli_table, coeffs=new_coeffs))
    qubits_updated = qubits_updated.reduce()
    return qubits_updated


def _calc_updated_coeffs(h: Union[PauliSumOp, PauliOp], table_z):
    coeffs = np.copy(h.primitive.coeffs[0])
    if len(table_z) > 1 and table_z[1] == np.bool_(True):
        coeffs = -1 * coeffs
    if len(table_z) > 6 and table_z[5] == np.bool_(True):
        coeffs = -1 * coeffs
    return coeffs


def _preset_binary_vals(table_z):
    for index in (0, 1, 2, 3, 5):
        _preset_single_binary_val(table_z, index)


def _preset_single_binary_val(table_z, index: int):
    try:
        table_z[index] = np.bool_(False)
    except IndexError:
        pass
