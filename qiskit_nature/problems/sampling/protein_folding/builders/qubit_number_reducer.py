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
from typing import Union, List, Dict

import numpy as np
from qiskit.opflow import PauliSumOp, PauliOp
from qiskit.quantum_info import PauliTable, SparsePauliOp, Pauli


def _remove_unused_qubits(total_hamiltonian: Union[PauliSumOp, PauliOp]):
    unused_qubits = _find_unused_qubits(total_hamiltonian)
    num_qubits = total_hamiltonian.num_qubits
    new_tables = []
    new_coeffs = []
    if isinstance(total_hamiltonian, PauliOp):
        table_z = total_hamiltonian.primitive.z
        table_x = total_hamiltonian.primitive.x
        new_table_z, new_table_x = _update_pauli_tables(num_qubits, table_x, table_z, unused_qubits)
        return PauliOp(Pauli((new_table_z, new_table_x)))

    elif isinstance(total_hamiltonian, PauliSumOp):
        for term in total_hamiltonian:
            table_z = term.primitive.table.Z[0]
            table_x = term.primitive.table.X[0]
            coeffs = term.primitive.coeffs[0]
            new_table_z, new_table_x = _update_pauli_tables(num_qubits, table_x, table_z,
                                                            unused_qubits)
            new_table = np.concatenate((new_table_x, new_table_z), axis=0)
            new_tables.append(new_table)
            new_coeffs.append(coeffs)
    new_pauli_table = PauliTable(data=new_tables)
    qubits_updated = PauliSumOp(SparsePauliOp(data=new_pauli_table, coeffs=new_coeffs))
    return qubits_updated.reduce()


def _update_pauli_tables(num_qubits: int, table_x, table_z, unused_qubits):
    new_table_z = []
    new_table_x = []
    for ind in range(num_qubits):
        if ind not in unused_qubits:
            new_table_z.append(table_z[ind])
            new_table_x.append(table_x[ind])

    return new_table_z, new_table_x


def _find_unused_qubits(total_hamiltonian: Union[PauliSumOp, PauliOp]) -> List[int]:
    used_map = {}
    unused = []
    num_qubits = total_hamiltonian.num_qubits
    if isinstance(total_hamiltonian, PauliOp):
        table_z = total_hamiltonian.primitive.z
        _update_used_map(num_qubits, table_z, used_map)

    elif isinstance(total_hamiltonian, PauliSumOp):
        for term in total_hamiltonian:
            table_z = term.primitive.table.Z[0]
            _update_used_map(num_qubits, table_z, used_map)

    for ind in range(num_qubits):
        if ind not in used_map.keys():
            unused.append(ind)

    return unused


def _update_used_map(num_qubits: int, table_z, used_map: Dict[int, bool]):
    for ind in range(num_qubits):
        if table_z[ind]:
            used_map[ind] = True
