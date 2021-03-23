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

"""The Linear Mapper."""

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit_nature.operators.second_quantization.spin_op import SpinOp
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_nature.operators.second_quantization import SecondQuantizedOp

from .spin_mapper import SpinMapper
from .qubit_mapper import QubitMapper
from ... import QiskitNatureError
from typing import List, Tuple


class LinearMapper(SpinMapper):
    """The Linear spin-to-qubit mapping. """

    def map(self, second_q_op: SpinOp) -> PauliSumOp:

        nmodes = second_q_op.register_length

        pauli_table = []
        for i in range(nmodes):
            a_z = np.asarray([0] * i + [0] + [0] * (nmodes - i - 1), dtype=bool)
            a_x = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
            b_z = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
            b_x = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)

            pauli_table.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))

        return self.mode_based_mapping(second_q_op, pauli_table)

    @staticmethod
    def mode_based_mapping(second_q_op: SecondQuantizedOp,
                           pauli_table: List[Tuple[Pauli, Pauli]]) -> PauliSumOp:
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
        # nmodes = len(pauli_table)
        #
        # if nmodes != second_q_op.register_length:
        #     raise QiskitNatureError(f"Pauli table len {nmodes} does not match"
        #                             f"operator register length {second_q_op.register_length}")
        #
        # all_false = np.asarray([False] * nmodes, dtype=bool)
        #
        # ret_op_list = []
        #
        # for label, coeff in second_q_op.to_list():
        #
        #     ret_op = SparsePauliOp(Pauli((all_false, all_false)), coeffs=[coeff])
        #
        #     for char in label.split(' '):
        #         ixyz, rest = char.split('_')
        #         rest = rest.split('^')
        #         position = int(rest[0])
        #
        #         power = None
        #         if len(rest) == 2:
        #             power = rest[1]
        #
        #         print(ixyz, position, power)
        #
        # zero_op = SparsePauliOp(Pauli((all_false, all_false)), coeffs=[0])
        #return PauliSumOp(sum(ret_op_list, zero_op)).reduce()

        raise NotImplementedError()

    def _linear_encoding(self, spin):
        """
        Generates a 'linear_encoding' of the spin S operators 'X', 'Y', 'Z' and 'identity'
        to qubit operators (linear combinations of pauli strings).
        In this 'linear_encoding' each individual spin S system is represented via
        2S+1 qubits and the state |s> is mapped to the state |00...010..00>, where the s-th qubit is
        in state 1.
        Returns:
            self.transformed_XYZI: list,
                The 4-element list of transformed spin S 'X', 'Y', 'Z' and 'identity' operators.
                I.e.
                    self.transformed_XYZI[0] corresponds to the linear combination of pauli strings needed
                    to represent the embedded 'X' operator
        """
        print('Linear encoding is calculated.')

        S = spin

        transformed_XYZI = []
        dim_S = int(2 * S + 1)
        nqubits = dim_S

        # quick functions to generate a pauli with X / Y / Z at location `i`
        pauli_id = Pauli.from_label('I' * nqubits)
        pauli_x = lambda i: Pauli.from_label('I' * i + 'X' + 'I' * (nqubits - i - 1))
        pauli_y = lambda i: Pauli.from_label('I' * i + 'Y' + 'I' * (nqubits - i - 1))
        pauli_z = lambda i: Pauli.from_label('I' * i + 'Z' + 'I' * (nqubits - i - 1))

        # 1. build the non-diagonal X operator
        x_summands = []
        for i, coeff in enumerate(np.diag(SpinOp("X", spin=S).to_matrix(), 1)):
            x_summands.append(PauliSumOp(coeff / 2. * SparsePauliOp(pauli_x(i) * pauli_x(i + 1)) +
                                         coeff / 2. * SparsePauliOp(pauli_y(i) * pauli_y(i + 1))))
        transformed_XYZI.append(self._operator_sum(x_summands))

        # 2. build the non-diagonal Y operator
        y_summands = []
        for i, coeff in enumerate(np.diag(SpinOp("Y", spin=S).to_matrix(), 1)):
            y_summands.append(PauliSumOp(-1j * coeff / 2. * SparsePauliOp(pauli_x(i) * pauli_y(i + 1)) +
                                         1j * coeff / 2. * SparsePauliOp(pauli_y(i) * pauli_x(i + 1))))
        transformed_XYZI.append(self._operator_sum(y_summands))

        # 3. build the diagonal Z
        z_summands = []
        for i, coeff in enumerate(np.diag(SpinOp("Z", spin=S).to_matrix())):  # get the first upper diagonal of coeff.
            z_summands.append(PauliSumOp(coeff / 2. * SparsePauliOp(pauli_z(i)) +
                                         coeff / 2. * SparsePauliOp(pauli_id)))
        z_operator = self._operator_sum(z_summands)
        transformed_XYZI.append(z_operator)

        # 4. add the identity operator
        transformed_XYZI.append(PauliSumOp(1. * SparsePauliOp(pauli_id)))

        # return the lookup table for the transformed XYZI operators
        return transformed_XYZI

    def _operator_sum(op_list):
        """Calculates the sum of all elements of a non-empty list
        Args:
            op_list (list):
                The list of objects to sum, i.e. [obj1, obj2, ..., objN]
        Returns:
            obj1 + obj2 + ... + objN
        """
        assert len(op_list) > 0, 'Operator list must be non-empty'

        if len(op_list) == 1:
            return copy.deepcopy(op_list[0])
        else:
            op_sum = copy.deepcopy(op_list[0])
            for elem in op_list[1:]:
                op_sum += elem
        return op_sum

    def _operator_product(op_list):
        """
        Calculates the product of all elements in a non-empty list.
        Args:
            op_list (list):
                The list of objects to sum, i.e. [obj1, obj2, ..., objN]
        Returns:
            obj1 * obj2 * ... * objN
        """
        assert len(op_list) > 0, 'Operator list must be non-empty'

        if len(op_list) == 1:
            return op_list[0]
        else:
            op_prod = op_list[0]
            for elem in op_list[1:]:
                op_prod *= elem
        return op_prod
