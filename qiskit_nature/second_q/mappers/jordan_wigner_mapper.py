# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
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


class JordanWignerMapper(FermionicMapper):
    """The Jordan-Wigner fermion-to-qubit mapping."""

    @classmethod
    @lru_cache(maxsize=32)
    def pauli_table(cls, register_length: int) -> list[tuple[Pauli, Pauli]]:
        # pylint: disable=unused-argument
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
        ``JordanWignerMapper`` to ensure accurate results.

        Args:
            qubit_op: The qubit operator ``SparsePauliOp`` to be mapped.

        Returns:
            The second quantized operator ``FermionicOp`` corresponding to
            the Hamiltonian in the Fermionic space.
        """
        num_qubits = (
            qubit_op.num_qubits
        )  # get number of qubits from input second quantized operator
        qubit_op = cls.__invert_pauli_terms(qubit_op)
        total_fermionic_op = FermionicOp.zero()
        for term in qubit_op:
            coef_term = term.coeffs[0]
            target_pauli_op = term.paulis[0]
            ferm_term_ops = []
            for i in range(num_qubits):
                one_pauli = target_pauli_op[num_qubits - 1 - i]
                pauli_char = one_pauli.to_label()
                if pauli_char == "Z":  # dealing Pauli Z op
                    ferm_op_pauli = FermionicOp({"": 1, f"+_{i} -_{i}": -2})
                elif pauli_char == "X":  # dealing Pauli X op
                    ferm_op_pauli = FermionicOp({f"+_{i}": 1, f"-_{i}": 1})
                    target_pauli_op = Pauli("I" * (i + 1) + "Z" * (num_qubits - i - 1)).compose(
                        target_pauli_op
                    )
                elif one_pauli.to_label() == "Y":  # dealing Pauli Y op
                    ferm_op_pauli = FermionicOp({f"+_{i}": -1j, f"-_{i}": 1j})
                    target_pauli_op = Pauli("I" * (i + 1) + "Z" * (num_qubits - i - 1)).compose(
                        target_pauli_op
                    )
                else:
                    ferm_op_pauli = FermionicOp.one()
                ferm_term_ops.append(ferm_op_pauli)
            term_fermionic_op = FermionicOp.one()
            for op in ferm_term_ops:
                term_fermionic_op = term_fermionic_op @ op
            if target_pauli_op.phase == 1:
                coef_term *= -1j
            elif target_pauli_op.phase == 2:
                coef_term *= -1
            elif target_pauli_op.phase == 3:
                coef_term *= 1j
            total_fermionic_op += coef_term * term_fermionic_op
        return total_fermionic_op.normal_order()

    @staticmethod
    def __invert_pauli_terms(sparse_pauli_op: SparsePauliOp) -> SparsePauliOp:
        """Utility to invert the order of Pauli operators in each term of a SparsePauliOp."""
        inverted_labels = [label[::-1] for label in sparse_pauli_op.paulis.to_labels()]
        # Create a new SparsePauliOp with the inverted labels but same coefficients
        inverted_sparse_pauli_op = SparsePauliOp(inverted_labels, sparse_pauli_op.coeffs)
        return inverted_sparse_pauli_op
