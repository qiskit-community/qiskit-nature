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
"""Removes qubit registers that are not relevant for the problem."""
from typing import Union, List, Dict, Tuple

import numpy as np
from qiskit.opflow import PauliSumOp, PauliOp, OperatorBase
from qiskit.quantum_info import PauliTable, SparsePauliOp, Pauli


def _remove_unused_qubits(
    total_hamiltonian: Union[PauliSumOp, PauliOp]
) -> Tuple[Union[PauliSumOp, PauliOp], List[int]]:
    """
    Removes those qubits from a total Hamiltonian that are equal to an identity operator across
    all terms, i.e. they are irrelevant for the problem. It makes the number of qubits required
    for encoding the problem smaller or equal.

    Args:
        total_hamiltonian: A full Hamiltonian for the protein folding problem.

    Returns:
        Tuple consisting of the total_hamiltonian compressed to an equivalent Hamiltonian and
        indices of qubits in the original Hamiltonian that were unused as optimization variables.
    """
    unused_qubits = _find_unused_qubits(total_hamiltonian)
    num_qubits = total_hamiltonian.num_qubits
    if isinstance(total_hamiltonian, PauliOp):
        return _compress_pauli_op(num_qubits, total_hamiltonian, unused_qubits), unused_qubits

    elif isinstance(total_hamiltonian, PauliSumOp):
        return _compress_pauli_sum_op(num_qubits, total_hamiltonian, unused_qubits), unused_qubits
    return None, None


def _compress_pauli_op(
    num_qubits: int,
    total_hamiltonian: Union[PauliSumOp, PauliOp, OperatorBase],
    unused_qubits: List[int],
) -> Union[PauliOp, OperatorBase]:
    table_z = total_hamiltonian.primitive.z
    table_x = total_hamiltonian.primitive.x
    new_table_z, new_table_x = _calc_reduced_pauli_tables(
        num_qubits, table_x, table_z, unused_qubits
    )
    total_hamiltonian_compressed = PauliOp(Pauli((new_table_z, new_table_x)))
    return total_hamiltonian_compressed


def _compress_pauli_sum_op(
    num_qubits: int,
    total_hamiltonian: Union[PauliSumOp, PauliOp, OperatorBase],
    unused_qubits: List[int],
) -> Union[PauliSumOp, PauliOp, OperatorBase]:
    new_tables = []
    new_coeffs = []
    for term in total_hamiltonian:
        table_z = term.primitive.paulis.z[0]
        table_x = term.primitive.paulis.x[0]
        coeffs = term.primitive.coeffs[0]
        new_table_z, new_table_x = _calc_reduced_pauli_tables(
            num_qubits, table_x, table_z, unused_qubits
        )
        new_table = np.concatenate((new_table_x, new_table_z), axis=0)
        new_tables.append(new_table)
        new_coeffs.append(coeffs)
    new_pauli_table = PauliTable(data=new_tables)
    total_hamiltonian_compressed = PauliSumOp(
        SparsePauliOp(data=new_pauli_table, coeffs=new_coeffs)
    ).reduce()
    return total_hamiltonian_compressed


def _calc_reduced_pauli_tables(
    num_qubits: int, table_x, table_z, unused_qubits
) -> Tuple[List[bool], List[bool]]:
    new_table_z = []
    new_table_x = []
    for ind in range(num_qubits):
        if ind not in unused_qubits:
            new_table_z.append(table_z[ind])
            new_table_x.append(table_x[ind])

    return new_table_z, new_table_x


def _find_unused_qubits(total_hamiltonian: Union[PauliSumOp, PauliOp]) -> List[int]:
    used_map: Dict[int, bool] = {}
    unused = []
    num_qubits = total_hamiltonian.num_qubits
    if isinstance(total_hamiltonian, PauliOp):
        table_z = total_hamiltonian.primitive.z
        _update_used_map(num_qubits, table_z, used_map)

    elif isinstance(total_hamiltonian, PauliSumOp):
        for term in total_hamiltonian:
            table_z = term.primitive.paulis.z[0]
            _update_used_map(num_qubits, table_z, used_map)

    for ind in range(num_qubits):
        if ind not in used_map.keys():
            unused.append(ind)

    return unused


def _update_used_map(num_qubits: int, table_z, used_map: Dict[int, bool]):
    for ind in range(num_qubits):
        if table_z[ind]:
            used_map[ind] = True
