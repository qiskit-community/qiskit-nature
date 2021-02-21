# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qubit Mapping interface."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from qiskit.aqua.operators import PauliSumOp
from qiskit.quantum_info.operators import Pauli, SparsePauliOp

from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization.particle_op import ParticleOp


class QubitMapping(ABC):
    """The interface for implementing methods which map from a `ParticleOp` to a
    `PauliSumOp`.
    """

    @abstractmethod
    def supports_particle_type(self, particle_type: ParticleOp) -> bool:
        """Returns whether the queried particle-type operator is supported by this mapping.

        Args:
            particle_type: the particle-type to query support for.

        Returns:
            A boolean indicating whether the queried particle-type is supported.
        """
        raise NotImplementedError()

    @abstractmethod
    def map(self, second_q_op: ParticleOp) -> PauliSumOp:
        """Maps a `ParticleOp` to a `PauliSumOp`.

        Args:
            second_q_op: the `ParticleOp` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.
        """
        raise NotImplementedError()

    @staticmethod
    def mode_based_mapping(second_q_op: ParticleOp,
                           pauli_table: List[Tuple[Pauli, Pauli]]) -> PauliSumOp:
        """Utility method to map a `ParticleOp` to a `PauliSumOp` using a pauli table.

        Args:
            second_q_op: the `ParticleOp` to be mapped.
            pauli_table: a table of paulis built according to the modes of the operator

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.

        Raises:
            QiskitNatureError: If number length of pauli table does not match the number
                of operator modes, or if the operator has unexpected label content
        """
        nmodes = len(pauli_table)
        if nmodes != second_q_op.register_length:
            raise QiskitNatureError(f"Pauli table len {nmodes} does not match"
                                    f"operator register length {second_q_op.register_length}")

            # 0. Some utilities
        def times_creation_op(op, position, pauli_table):
            # The creation operator is given by 0.5*(X + 1j*Y)
            real_part = SparsePauliOp(pauli_table[position][0], coeffs=[0.5])
            imag_part = SparsePauliOp(pauli_table[position][1], coeffs=[0.5j])

            # We must multiply from the left due to the right-to-left execution order of operators.
            prod = (real_part + imag_part) * op
            return prod

        def times_annihilation_op(op, position, pauli_table):
            # The annihilation operator is given by 0.5*(X - 1j*Y)
            real_part = SparsePauliOp(pauli_table[position][0], coeffs=[0.5])
            imag_part = SparsePauliOp(pauli_table[position][1], coeffs=[-0.5j])

            # We must multiply from the left due to the right-to-left execution order of operators.
            prod = (real_part + imag_part) * op
            return prod

        # 1. Initialize an operator list with the identity scaled by the `self.coeff`
        all_false = np.asarray([False] * nmodes, dtype=bool)

        ret_op_list = []

        # TODO to_list() is not an attribute of ParticleOp. Change the former to have this or
        #   change the signature above to take FermionicOp?
        #
        for label, coeff in second_q_op.to_list():

            ret_op = SparsePauliOp(Pauli((all_false, all_false)), coeffs=[coeff])

            # Go through the label and replace the fermion operators by their qubit-equivalent, then
            # save the respective Pauli string in the pauli_str list.
            for position, char in enumerate(label):
                if char == "+":
                    ret_op = times_creation_op(ret_op, position, pauli_table)
                elif char == "-":
                    ret_op = times_annihilation_op(ret_op, position, pauli_table)
                elif char == "N":
                    # The occupation number operator N is given by `+-`.
                    ret_op = times_creation_op(ret_op, position, pauli_table)
                    ret_op = times_annihilation_op(ret_op, position, pauli_table)
                elif char == "E":
                    # The `emptiness number` operator E is given by `-+` = (I - N).
                    ret_op = times_annihilation_op(ret_op, position, pauli_table)
                    ret_op = times_creation_op(ret_op, position, pauli_table)
                elif char == "I":
                    continue

                # catch any disallowed labels
                else:
                    raise QiskitNatureError(
                        f"BaseFermionOperator label included '{char}'. "
                        "Allowed characters: I, N, E, +, -"
                    )
            ret_op_list.append(ret_op)

        zero_op = SparsePauliOp(Pauli((all_false, all_false)), coeffs=[0])
        return PauliSumOp(sum(ret_op_list, zero_op)).reduce()
