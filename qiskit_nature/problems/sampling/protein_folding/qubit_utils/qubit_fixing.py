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
"""Changes certain qubits to fixed values."""
from typing import Union

import numpy as np
from qiskit.opflow import PauliSumOp, OperatorBase, PauliOp
from qiskit.quantum_info import PauliTable, SparsePauliOp, Pauli


def _fix_qubits(
    operator: Union[int, PauliSumOp, PauliOp, OperatorBase],
    has_side_chain_second_bead: bool = False,
) -> Union[int, PauliSumOp, PauliOp, OperatorBase]:
    """
    Assigns predefined values for turns qubits on positions 0, 1, 2, 3, 5 in the main chain
    without the loss of generality (see the paper https://arxiv.org/pdf/1908.02163.pdf). Qubits
    on these position are considered fixed and not subject to optimization.

    Args:
        operator: an operator whose qubits shall be fixed.

    Returns:
        An operator with relevant qubits changed to fixed values.
    """
    # operator might be 0 (int) because it is initialized as operator = 0; then we should not
    # attempt fixing qubits
    if (
        not isinstance(operator, PauliOp)
        and not isinstance(operator, PauliSumOp)
        and not isinstance(operator, OperatorBase)
    ):
        return operator
    operator = operator.reduce()
    new_tables = []
    new_coeffs = []
    if isinstance(operator, PauliOp):
        table_z = np.copy(operator.primitive.z)
        table_x = np.copy(operator.primitive.x)
        _preset_binary_vals(table_z, has_side_chain_second_bead)
        return PauliOp(Pauli((table_z, table_x)))

    for hamiltonian in operator:
        table_z = np.copy(hamiltonian.primitive.paulis.z[0])
        table_x = np.copy(hamiltonian.primitive.paulis.x[0])
        coeffs = _calc_updated_coeffs(hamiltonian, table_z, has_side_chain_second_bead)
        _preset_binary_vals(table_z, has_side_chain_second_bead)
        new_table = np.concatenate((table_x, table_z), axis=0)
        new_tables.append(new_table)
        new_coeffs.append(coeffs)
    new_pauli_table = PauliTable(data=new_tables)
    operator_updated = PauliSumOp(SparsePauliOp(data=new_pauli_table, coeffs=new_coeffs))
    operator_updated = operator_updated.reduce()
    return operator_updated


def _calc_updated_coeffs(
    hamiltonian: Union[PauliSumOp, PauliOp], table_z, has_side_chain_second_bead: bool
) -> np.ndarray:
    coeffs = np.copy(hamiltonian.primitive.coeffs[0])
    if len(table_z) > 1 and table_z[1] == np.bool_(True):
        coeffs = -1 * coeffs
    if not has_side_chain_second_bead and len(table_z) > 6 and table_z[5] == np.bool_(True):
        coeffs = -1 * coeffs
    return coeffs


def _preset_binary_vals(table_z, has_side_chain_second_bead: bool):
    main_beads_indices = [0, 1, 2, 3]
    if not has_side_chain_second_bead:
        main_beads_indices.append(5)
    for index in main_beads_indices:
        _preset_single_binary_val(table_z, index)


def _preset_single_binary_val(table_z, index: int):
    try:
        table_z[index] = np.bool_(False)
    except IndexError:
        pass
