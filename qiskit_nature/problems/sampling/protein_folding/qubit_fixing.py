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


def _fix_qubits(qubits):
    new_tables = []
    new_coeffs = []
    for i in range(len(qubits)):
        H = qubits[i]
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
    qubits_updated = PauliSumOp(SparsePauliOp(data=new_pauli_table, coeffs=new_coeffs))
    qubits_updated = qubits_updated.reduce()
    return qubits_updated