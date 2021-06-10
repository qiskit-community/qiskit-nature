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
from qiskit.quantum_info import PauliTable, SparsePauliOp


def _fix_qubits(qubits: PauliSumOp):
    new_tables = []
    new_coeffs = []
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


def _calc_updated_coeffs(h: PauliSumOp, table_z):
    coeffs = np.copy(h.primitive.coeffs[0])
    if table_z[1] == np.bool_(True):
        coeffs = -1 * coeffs
    if table_z[5] == np.bool_(True):
        coeffs = -1 * coeffs
    return coeffs


def _preset_binary_vals(table_z):
    table_z[0] = np.bool_(False)
    table_z[1] = np.bool_(False)
    table_z[2] = np.bool_(False)
    table_z[3] = np.bool_(False)
    table_z[5] = np.bool_(False)