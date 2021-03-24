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

import copy

from fractions import Fraction
from typing import List, Union
import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_nature.operators.second_quantization.spin_op import SpinOp
from .spin_mapper import SpinMapper


class LinearMapper(SpinMapper):
    """The Linear spin-to-qubit mapping. """

    def map(self, second_q_op: SpinOp) -> PauliSumOp:

        qubit_ops_list = []

        # get transformed general spin matrices
        spinx, spiny, spinz, identity = self._linear_encoding(second_q_op.spin)

        for _, op in enumerate(second_q_op.to_list()):

            operatorlist = []

            oper, coeff = op

            for n_x, n_y, n_z in zip(second_q_op.x[_], second_q_op.y[_], second_q_op.z[_]):

                operator_on_spin_i = []

                if n_x > 0:
                    # construct the qubit operator embed
                    operator_on_spin_i.append(
                        self._operator_product([spinx for i in range(int(n_x))]))

                if n_y > 0:
                    # construct the qubit operator embed
                    operator_on_spin_i.append(
                        self._operator_product([spiny for i in range(int(n_y))]))

                if n_z > 0:
                    # construct the qubit operator embed
                    operator_on_spin_i.append(
                        self._operator_product([spinz for i in range(int(n_z))]))

                if np.any([n_x, n_y, n_z]) > 0:
                    # multiply X^n_x * Y^n_y * Z^n_z
                    operator_on_spin_i = self._operator_product(operator_on_spin_i)
                    operatorlist.append(operator_on_spin_i.reduce())

                else:
                    # If n_x=n_y=n_z=0, simply add the embedded Identity operator.
                    operatorlist.append(identity)

                # A list which still need to tensor together
                # to get the final operator
                # first we reduce operators

                tmp_operatorlist = []

                for tmp_op in operatorlist:
                    print(type(tmp_op))
                    tmp_operatorlist.append(tmp_op.reduce())

                operatorlist = tmp_operatorlist
            assert False
            qubit_ops_list.append(self._tensor_ops(operatorlist, coeff))

        qubit_op = self._operator_sum(qubit_ops_list)

        return qubit_op

    def _operator_sum(self, op_list: List) -> PauliSumOp:
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

    def _operator_product(self, op_list) -> PauliSumOp:
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

    def _linear_encoding(self, spin: Union[Fraction, float]) -> List:
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
                    self.transformed_XYZI[0] corresponds to the linear
                    combination of pauli strings needed
                    to represent the embedded 'X' operator
        """

        trafo_xzyi = []
        dspin = int(2 * spin + 1)
        nqubits = dspin

        # quick functions to generate a pauli with X / Y / Z at location `i`
        pauli_id = Pauli.from_label('I' * nqubits)

        def pauli_x(i):
            return Pauli.from_label('I' * i + 'X' + 'I' * (nqubits - i - 1))

        def pauli_y(i):
            return Pauli.from_label('I' * i + 'Y' + 'I' * (nqubits - i - 1))

        def pauli_z(i):
            return Pauli.from_label('I' * i + 'Z' + 'I' * (nqubits - i - 1))

        # 1. build the non-diagonal X operator
        x_summands = []
        for i, coeff in enumerate(np.diag(SpinOp("X", spin=spin).to_matrix(), 1)):
            x_summands.append(PauliSumOp(coeff / 2. * SparsePauliOp(pauli_x(i) * pauli_x(i + 1)) +
                                         coeff / 2. * SparsePauliOp(pauli_y(i) * pauli_y(i + 1))))
        trafo_xzyi.append(self._operator_sum(x_summands))

        # 2. build the non-diagonal Y operator
        y_summands = []
        for i, coeff in enumerate(np.diag(SpinOp("Y", spin=spin).to_matrix(), 1)):
            y_summands.append(PauliSumOp(-1j * coeff / 2. *
                                         SparsePauliOp(pauli_x(i) * pauli_y(i + 1)) +
                                         1j * coeff / 2. *
                                         SparsePauliOp(pauli_y(i) * pauli_x(i + 1))))
        trafo_xzyi.append(self._operator_sum(y_summands))

        # 3. build the diagonal Z
        z_summands = []
        for i, coeff in enumerate(np.diag(SpinOp("Z", spin=spin).to_matrix())):
            # get the first upper diagonal of coeff.
            z_summands.append(PauliSumOp(coeff / 2. * SparsePauliOp(pauli_z(i)) +
                                         coeff / 2. * SparsePauliOp(pauli_id)))

        z_operator = self._operator_sum(z_summands)
        trafo_xzyi.append(z_operator)

        # 4. add the identity operator
        trafo_xzyi.append(PauliSumOp(1. * SparsePauliOp(pauli_id)))

        # return the lookup table for the transformed XYZI operators
        return trafo_xzyi

    def _tensor_ops(self, operatorlist: List, coeff: complex) -> PauliSumOp:

        if len(operatorlist) == 1:
            tensor_op = operatorlist[0]

        elif len(operatorlist) > 1:
            tensor_op = operatorlist[0]

            for op in operatorlist[1:]:
                tensor_op = tensor_op ^ op
        else:
            raise 'Unsupported list provided'

        return coeff * tensor_op
