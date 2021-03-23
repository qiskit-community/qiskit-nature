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
from typing import List, Optional, Union
from fractions import Fraction
import copy

from qiskit.opflow import PauliSumOp
from qiskit_nature.operators.second_quantization.spin_op import SpinOp
from qiskit.quantum_info.operators import Pauli, SparsePauliOp

from .spin_mapper import SpinMapper

class LinearMapper(SpinMapper):
    """The Linear spin-to-qubit mapping. """

    def map(self, second_q_op: SpinOp) -> PauliSumOp:

        qubit_ops_list = []

        # get transformed generalised spin matrices transformed_xyzi
        spinx, spiny, spinz, identity = self._linear_encoding(second_q_op.spin)

        for _, op in enumerate(second_q_op.to_list()):

            operatorlist = []

            oper, coeff = op

            for nx, ny, nz in zip(second_q_op.x[_], second_q_op.y[_], second_q_op.z[_]):

                operator_on_spin_i = []

                if nx > 0:
                    # construct the qubit operator embed(X^nx)
                    operator_on_spin_i.append(self._operator_product([spinx for i in range(int(nx))]))

                if ny > 0:
                    # construct the qubit operator embed(Y^ny)
                    operator_on_spin_i.append(self._operator_product([spiny for i in range(int(ny))]))

                if nz > 0:
                    # construct the qubit operator embed(Z^nz)
                    operator_on_spin_i.append(self._operator_product([spinz for i in range(int(nz))]))

                if np.any([nx, ny, nz]) > 0:
                    # multiply X^nx * Y^ny * Z^nz
                    operator_on_spin_i = self._operator_product(operator_on_spin_i)
                    operatorlist.append(operator_on_spin_i)

                else:
                    # If nx=ny=nz=0, simply add the embedded Identity operator.
                    operatorlist.append(identity)

                # `operatorlist` is now a list of (sums of pauli strings) which still need to be tensored together
                # to get the final operator
                # first we reduce operators

                operatorlist_red = []

                for op in operatorlist:
                    operatorlist_red.append(op.reduce())

                operatorlist = operatorlist_red

            qubit_ops_list.append(self._tensor_ops(operatorlist, coeff))

        qubit_op = self._operator_sum(qubit_ops_list)

        return qubit_op

    def _operator_sum(self, op_list):
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

    def _operator_product(self,op_list):
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
                op_prod = op_prod @ elem
        return op_prod

    def _linear_encoding(self, S: Union[Fraction, float]):
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

        #             WeightedPauliOperator(paulis=[[coeff/2., pauli_z(i)],
        #                                            [coeff/2., pauli_id]])
        #                           )
        z_operator = self._operator_sum(z_summands)
        transformed_XYZI.append(z_operator)

        # 4. add the identity operator
        transformed_XYZI.append(PauliSumOp(1. * SparsePauliOp(pauli_id)))

        # return the lookup table for the transformed XYZI operators
        return transformed_XYZI

    def _tensor_ops(self, operatorlist: List, coeff: complex):

        if len(operatorlist) == 1:
            tensored_op = operatorlist[0]

        elif len(operatorlist) > 1:
            tensored_op = operatorlist[0]

            for op in operatorlist[1:]:
                tensored_op = tensored_op ^ op
        else:
            raise 'Unsupported list provided'

        return coeff * tensored_op
