# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Logarithmic Mapper for Bosons."""

from __future__ import annotations
import operator
import math

from functools import reduce, lru_cache

import numpy as np

from qiskit.quantum_info import Pauli, SparsePauliOp

from qiskit_nature.second_q.operators import BosonicOp
from .bosonic_mapper import BosonicMapper

class BosonicLogarithmicMapper(BosonicMapper):
    """
    The Bosonic Logarithmic Mapper.
    """
    def __init__(self, max_occupation: int) -> None:
        # Compute the actual max occupation from the one selected by the user
        self.number_of_qubits_per_mode: int = 1 if max_occupation == 0 else math.ceil(np.log2(max_occupation + 1))
        max_occupation = 2**self.number_of_qubits_per_mode - 1
        super().__init__(max_occupation)

    @property
    def number_of_qubits_per_mode(self) -> int:
        """The minimum number of qubits required to  of any bosonic state."""
        return self._number_of_qubits_per_mode

    @number_of_qubits_per_mode.setter
    def number_of_qubits_per_mode(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError(
                f"The number of qubits must be at least 1, and not {num_qubits}."
            )
        self._number_of_qubits_per_mode: int = num_qubits

    def _map_single(self, second_q_op: BosonicOp, *, register_length: int | None = None) -> SparsePauliOp:
        """Maps a :class:`~qiskit_nature.second_q.operators.SparseLabelOp` to a ``SparsePauliOp``.

        Args:
            second_q_op: the ``SparseLabelOp`` to be mapped.
            register_length: when provided, this will be used to overwrite the ``register_length``
                attribute of the operator being mapped. This is possible because the
                ``register_length`` is considered a lower bound in a ``SparseLabelOp``.

        Returns:
            The qubit operator corresponding to the problem-Hamiltonian in the qubit space.
        """
        if register_length is None:
            register_length = second_q_op.num_modes

        # The actual register length is the number of qubits per mode times the number of modes
        qubit_register_length: int = register_length * self.number_of_qubits_per_mode
        # Create a Pauli operator, which we will fill in this method
        pauli_op: list[SparsePauliOp] = []
        # Then we loop over all the terms of the bosonic operator
        for terms, coeff in second_q_op.terms():
            # Then loop over each term (terms -> List[Tuple[string, int]])
            bos_op_to_pauli_op = SparsePauliOp(["I" * qubit_register_length], coeffs=[1.0])
            for op, idx in terms:
                if op not in ("+", "-"):
                    break
                pauli_expansion: list[SparsePauliOp] = []
                mode_index_in_register: int = idx * (self.number_of_qubits_per_mode)
                terms_idx = range(2**self.number_of_qubits_per_mode - 1) if op == "+" else range(1, 2**self.number_of_qubits_per_mode)
                for n in terms_idx:
                    prefactor = np.sqrt(n + 1)
                    print(f"n: {n}, prefactor: {prefactor}")
                    # Define the initial and final states (which results from the action of the operator)
                    final_state: str = bin(n + 1).split("b")[1].rjust(self.number_of_qubits_per_mode, "0")
                    init_state: str = bin(n).split("b")[1].rjust(self.number_of_qubits_per_mode, "0")
                    print(f"final_state: {final_state}, init_state: {init_state}")
                    # At this point, we have the following situation: sqrt(n+1)*|n+1><n|, where the states are represented
                    # in binary. We have to convert this to the corresponding Pauli operators
                    # Now build the Pauli operators
                    single_mapped_term = SparsePauliOp(["I" * qubit_register_length], coeffs=[1.0])
                    for j in range(len(init_state)):# - 1, -1, -1):
                        i: int = len(init_state) - j - 1
                        print(f"i: {i}, op: {final_state[j]}{init_state[j]}")
                        # Case |0><0|: this should be converted to 0.5*(I + Z)
                        if f"{final_state[j]}{init_state[j]}" == "00":
                            single_mapped_term: SparsePauliOp = single_mapped_term.compose(
                                self._get_single_qubit_pauli_matrix(mode_index_in_register, qubit_register_length, i, "I+"))
                        # Case |1><1|: this should be converted to 0.5*(I - Z)
                        elif f"{final_state[j]}{init_state[j]}" == "11":
                            single_mapped_term: SparsePauliOp = single_mapped_term.compose(
                                self._get_single_qubit_pauli_matrix(mode_index_in_register, qubit_register_length, i, "I-"))
                        # Case |0><1|: this should be converted to 0.5*(X + iY)
                        elif f"{final_state[j]}{init_state[j]}" == "01":
                            single_mapped_term: SparsePauliOp = single_mapped_term.compose(
                                self._get_single_qubit_pauli_matrix(mode_index_in_register, qubit_register_length, i, "S+"))
                        # Case |1><0|: this should be converted to 0.5*(X - iY)
                        elif f"{final_state[j]}{init_state[j]}" == "10":
                            single_mapped_term: SparsePauliOp = single_mapped_term.compose(
                                self._get_single_qubit_pauli_matrix(mode_index_in_register, qubit_register_length, i, "S-"))
                        else:
                            raise ValueError(f"Invalid state {final_state[i]}{init_state[i]}.")
                    pauli_expansion.append(prefactor * single_mapped_term)
                # Add the Pauli expansion for a single n_k to map of the bosonic operator
                bos_op_to_pauli_op = reduce(operator.add, pauli_expansion).compose(
                    bos_op_to_pauli_op
                )
                print(f"bos_op_to_pauli_op: {bos_op_to_pauli_op}")
            # Add the map of the single boson op (e.g. +_0) to the map of the full bosonic operator
            pauli_op.append(coeff * reduce(operator.add, bos_op_to_pauli_op.simplify()))
            print(f"pauli_op: {reduce(operator.add, pauli_op)}")
        # return the lookup table for the transformed XYZI operators
        return reduce(operator.add, pauli_op)
    

    @lru_cache(maxsize=32)
    def _get_single_qubit_pauli_matrix(
        self, register_index: int, register_length: int, qubit_index: int, pauli_op: str
    ) -> SparsePauliOp:
        """This method builds the Qiskit Pauli operators of the operators
        I_+ = I + Z, I_- = I - Z, S_+ = X + iY and S_- = X - iY.

        Args:
            register_index: the index of the qubit register where the mapped operator should be placed.
            register_length: the length of the qubit register.

        Returns:
            Four Pauli operators that represent XX, XY, YX and YY at the specified index in the
            current qubit register.
        """
        # Define recurrent variables
        prefix_zeros = [0] * register_index + [0] * qubit_index
        suffix_zeros = ([0] * (register_length - self.number_of_qubits_per_mode - register_index) +
                                   [0] * (self.number_of_qubits_per_mode - qubit_index - 1))
        if pauli_op == "I+" or pauli_op == "I-":
            identity = Pauli(
                (
                    [0] * register_length,
                    [0] * register_length,
                )
            )
            sigma_z = Pauli(
                (
                    prefix_zeros + [1] + suffix_zeros,
                    [0] * register_length,
                )
            )
            if pauli_op == "I+":
                return 0.5 * (SparsePauliOp(identity) + SparsePauliOp(sigma_z))
            return 0.5 * (SparsePauliOp(identity) - SparsePauliOp(sigma_z))
        if pauli_op == "S+" or pauli_op == "S-":
            sigma_x = Pauli(
                (
                    [0] * register_length,
                    prefix_zeros + [1] + suffix_zeros,
                )
            )
            sigma_y = Pauli(
                (
                    prefix_zeros + [1] + suffix_zeros,
                    prefix_zeros + [1] + suffix_zeros,
                )
            )
            if pauli_op == "S+":
                return 0.5 * (SparsePauliOp(sigma_x) + 1j*SparsePauliOp(sigma_y))
            return 0.5 * (SparsePauliOp(sigma_x) - 1j*SparsePauliOp(sigma_y))
        raise ValueError(f"Invalid operator {pauli_op}. Possible values are 'I+', 'I-', 'S+' and 'S-'.")
