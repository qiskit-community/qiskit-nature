# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qubit Mapper interface."""

from abc import ABC, abstractmethod
from typing import List, Tuple

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli, SparsePauliOp

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators import FermionicOp, SecondQuantizedOp


class QubitMapper(ABC):
    """The interface for implementing methods which map from a `SecondQuantizedOp` to a
    qubit operator in the form of a `PauliSumOp`.
    """

    def __init__(self, allows_two_qubit_reduction: bool = False):
        """
        Args:
            allows_two_qubit_reduction: Set if mapper will create known symmetry such that the
                number of qubits in the mapped operator can be reduced accordingly.
        """
        self._allows_two_qubit_reduction = allows_two_qubit_reduction

    @property
    def allows_two_qubit_reduction(self) -> bool:
        """
        Getter for symmetry information for two qubit reduction

        Returns: If mapping generates the known symmetry that allows two qubit reduction.

        """
        return self._allows_two_qubit_reduction

    @abstractmethod
    def map(self, second_q_op: SecondQuantizedOp) -> PauliSumOp:
        """Maps a :class:`~qiskit_nature.second_q.operators.SecondQuantizedOp`
        to a `PauliSumOp`.

        Args:
            second_q_op: the `SecondQuantizedOp` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.
        """
        raise NotImplementedError()

    @staticmethod
    def mode_based_mapping(
        second_q_op: SecondQuantizedOp, pauli_table: List[Tuple[Pauli, Pauli]]
    ) -> PauliSumOp:
        """Utility method to map a `SecondQuantizedOp` to a `PauliSumOp` using a pauli table.

        Args:
            second_q_op: the `SecondQuantizedOp` to be mapped.
            pauli_table: a table of paulis built according to the modes of the operator

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.

        Raises:
            QiskitNatureError: If number length of pauli table does not match the number
                of operator modes, or if the operator has unexpected label content
        """
        nmodes = len(pauli_table)
        if nmodes != second_q_op.register_length:
            raise QiskitNatureError(
                f"Pauli table len {nmodes} does not match"
                f"operator register length {second_q_op.register_length}"
            )

        # 0. Some utilities

        times_creation_op = []
        times_annihilation_op = []
        times_occupation_number_op = []
        times_emptiness_number_op = []
        for paulis in pauli_table:
            real_part = SparsePauliOp(paulis[0], coeffs=[0.5])
            imag_part = SparsePauliOp(paulis[1], coeffs=[0.5j])

            # The creation operator is given by 0.5*(X - 1j*Y)
            creation_op = real_part - imag_part
            times_creation_op.append(creation_op)

            # The annihilation operator is given by 0.5*(X + 1j*Y)
            annihilation_op = real_part + imag_part
            times_annihilation_op.append(annihilation_op)

            # The occupation number operator N is given by `+-`.
            times_occupation_number_op.append(
                creation_op.compose(annihilation_op, front=True).simplify()
            )

            # The `emptiness number` operator E is given by `-+` = (I - N).
            times_emptiness_number_op.append(
                annihilation_op.compose(creation_op, front=True).simplify()
            )

        # make sure ret_op_list is not empty by including a zero op
        ret_op_list = [SparsePauliOp("I" * nmodes, coeffs=[0])]

        # TODO to_list() is not an attribute of SecondQuantizedOp. Change the former to have this or
        #   change the signature above to take FermionicOp?
        label_coeff_list = (
            second_q_op.to_list(display_format="dense")
            if isinstance(second_q_op, FermionicOp)
            else second_q_op.to_list()
        )
        for label, coeff in label_coeff_list:

            # 1. Initialize an operator list with the identity scaled by the `self.coeff`
            ret_op = SparsePauliOp("I" * nmodes, coeffs=[coeff])

            # Go through the label and replace the fermion operators by their qubit-equivalent, then
            # save the respective Pauli string in the pauli_str list.
            for position, char in enumerate(label):
                if char == "+":
                    ret_op = ret_op.compose(times_creation_op[position], front=True)
                elif char == "-":
                    ret_op = ret_op.compose(times_annihilation_op[position], front=True)
                elif char == "N":
                    ret_op = ret_op.compose(times_occupation_number_op[position], front=True)
                elif char == "E":
                    ret_op = ret_op.compose(times_emptiness_number_op[position], front=True)
                elif char == "I":
                    continue

                # catch any disallowed labels
                else:
                    raise QiskitNatureError(
                        f"FermionicOp label included '{char}'. Allowed characters: I, N, E, +, -"
                    )
            ret_op_list.append(ret_op)

        return PauliSumOp(SparsePauliOp.sum(ret_op_list).simplify())
