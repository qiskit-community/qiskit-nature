# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Jordan-Wigner Mapper. """

from __future__ import annotations

from functools import lru_cache

import numpy as np

from qiskit.quantum_info.operators import Pauli
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import FermionicOp
from .fermionic_mapper import FermionicMapper
from .mode_based_mapper import ModeBasedMapper, PauliType


class JordanWignerMapper(FermionicMapper, ModeBasedMapper):
    """The Jordan-Wigner fermion-to-qubit mapping."""

    def pauli_table(self, register_length: int) -> list[tuple[PauliType, PauliType]]:
        return self._pauli_table(register_length)

    @staticmethod
    @lru_cache(maxsize=32)
    def _pauli_table(register_length: int) -> list[tuple[PauliType, PauliType]]:
        pauli_table = []

        for i in range(register_length):
            a_z = np.asarray([1] * i + [0] + [0] * (register_length - i - 1), dtype=bool)
            a_x = np.asarray([0] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            b_z = np.asarray([1] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            b_x = np.asarray([0] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            # c_z = np.asarray([0] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            # c_x = np.asarray([0] * register_length, dtype=bool)
            pauli_table.append((Pauli((a_z, a_x)), Pauli((b_z, b_x))))
            # TODO add Pauli 3-tuple to lookup table
        return pauli_table

    @classmethod
    def reverse_map(cls, qubit_op: SparsePauliOp) -> FermionicOp:
        """Maps a qubit operator ``SparsePauliOp`` back into the second quantized
        operator ``FermionicOp``. While it'll provide an output for any ``SparsePauliOp``
        operator, it should be used on operators that were created with
        ``JordanWignerMapper`` to ensure a meaningful result.

        Args:
            qubit_op: The qubit operator ``SparsePauliOp`` to be mapped.

        Returns:
            The second quantized operator ``FermionicOp`` corresponding to
            the Hamiltonian in the Fermionic space.
        """
        num_qubits = qubit_op.num_qubits
        total_fermionic_op = FermionicOp({}, num_spin_orbitals=num_qubits)
        for target_pauli_op, coef_term in zip(qubit_op.paulis, qubit_op.coeffs):
            term_fermionic_op = FermionicOp({"": 1}, num_spin_orbitals=num_qubits)
            # X and Y introduce parity Zs on lower modes, so process higher modes first.
            for i in reversed(range(num_qubits)):
                pauli_char = target_pauli_op[i].to_label()
                if pauli_char == "Z":  # dealing Pauli Z op
                    ferm_op_pauli = FermionicOp({"": 1, f"+_{i} -_{i}": -2})
                elif pauli_char == "X":  # dealing Pauli X op
                    ferm_op_pauli = FermionicOp({f"+_{i}": 1, f"-_{i}": 1})
                elif pauli_char == "Y":  # dealing Pauli Y op
                    ferm_op_pauli = FermionicOp({f"+_{i}": 1j, f"-_{i}": -1j})
                else:
                    continue
                if pauli_char in ("X", "Y"):
                    parity_pauli = Pauli("I" * (num_qubits - i) + "Z" * i)
                    target_pauli_op = parity_pauli @ target_pauli_op
                term_fermionic_op = term_fermionic_op @ ferm_op_pauli
            coef_term *= (-1j) ** target_pauli_op.phase
            total_fermionic_op += coef_term * term_fermionic_op
        return total_fermionic_op.normal_order()
