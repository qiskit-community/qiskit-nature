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

"""The Linear Mapper."""

import operator

from fractions import Fraction
from functools import reduce
from typing import List, Union

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli, SparsePauliOp

from qiskit_nature.second_q.operators import SpinOp
from .spin_mapper import SpinMapper


class LinearMapper(SpinMapper):  # pylint: disable=missing-class-docstring
    def __init__(self):
        """The Linear spin-to-qubit mapping."""
        super().__init__(allows_two_qubit_reduction=False)

    def map(self, second_q_op: SpinOp) -> PauliSumOp:

        qubit_ops_list: List[PauliSumOp] = []

        # get linear encoding of the general spin matrices
        spinx, spiny, spinz, identity = self._linear_encoding(second_q_op.spin)

        for idx, (_, coeff) in enumerate(second_q_op.to_list()):

            operatorlist: List[PauliSumOp] = []

            for n_x, n_y, n_z in zip(second_q_op.x[idx], second_q_op.y[idx], second_q_op.z[idx]):

                operator_on_spin_i: List[PauliSumOp] = []

                if n_x > 0:
                    operator_on_spin_i.append(reduce(operator.matmul, [spinx] * int(n_x)))

                if n_y > 0:
                    operator_on_spin_i.append(reduce(operator.matmul, [spiny] * int(n_y)))

                if n_z > 0:
                    operator_on_spin_i.append(reduce(operator.matmul, [spinz] * int(n_z)))

                if np.any([n_x, n_y, n_z]) > 0:
                    single_operator_on_spin_i = reduce(operator.matmul, operator_on_spin_i)
                    operatorlist.append(single_operator_on_spin_i.reduce())

                else:
                    # If n_x=n_y=n_z=0, simply add the embedded Identity operator.
                    operatorlist.append(identity)

            # Now, we can tensor all operators in this list
            # NOTE: in Qiskit's opflow the `XOR` (i.e. `^`) operator does the tensor product
            qubit_ops_list.append(coeff * reduce(operator.xor, reversed(operatorlist)))

        qubit_op = reduce(operator.add, qubit_ops_list)

        return qubit_op

    def _linear_encoding(self, spin: Union[Fraction, float]) -> List[PauliSumOp]:
        """
        Generates a 'linear_encoding' of the spin S operators 'X', 'Y', 'Z' and 'identity'
        to qubit operators (linear combinations of pauli strings).
        In this 'linear_encoding' each individual spin S system is represented via
        2S+1 qubits and the state |s> is mapped to the state |00...010..00>, where the s-th qubit is
        in state 1.

        Returns:
            The 4-element list of transformed spin S 'X', 'Y', 'Z' and 'identity' operators.
            I.e. spin_op_encoding[0]` corresponds to the linear combination of pauli strings needed
            to represent the embedded 'X' operator
        """

        spin_op_encoding: List[PauliSumOp] = []
        dspin = int(2 * spin + 1)
        nqubits = dspin

        # quick functions to generate a pauli with X / Y / Z at location `i`
        pauli_id = Pauli("I" * nqubits)

        def pauli_x(i):
            return Pauli("I" * i + "X" + "I" * (nqubits - i - 1))

        def pauli_y(i):
            return Pauli("I" * i + "Y" + "I" * (nqubits - i - 1))

        def pauli_z(i):
            return Pauli("I" * i + "Z" + "I" * (nqubits - i - 1))

        # 1. build the non-diagonal X operator
        x_summands = []
        for i, coeff in enumerate(np.diag(SpinOp("X", spin=spin).to_matrix(), 1)):
            x_summands.append(
                PauliSumOp(
                    coeff / 2.0 * SparsePauliOp(pauli_x(i).dot(pauli_x(i + 1)))
                    + coeff / 2.0 * SparsePauliOp(pauli_y(i).dot(pauli_y(i + 1)))
                )
            )
        spin_op_encoding.append(reduce(operator.add, x_summands))

        # 2. build the non-diagonal Y operator
        y_summands = []
        for i, coeff in enumerate(np.diag(SpinOp("Y", spin=spin).to_matrix(), 1)):
            y_summands.append(
                PauliSumOp(
                    -1j * coeff / 2.0 * SparsePauliOp(pauli_x(i).dot(pauli_y(i + 1)))
                    + 1j * coeff / 2.0 * SparsePauliOp(pauli_y(i).dot(pauli_x(i + 1)))
                )
            )
        spin_op_encoding.append(reduce(operator.add, y_summands))

        # 3. build the diagonal Z
        z_summands = []
        for i, coeff in enumerate(np.diag(SpinOp("Z", spin=spin).to_matrix())):
            # get the first upper diagonal of coeff.
            z_summands.append(
                PauliSumOp(
                    coeff / 2.0 * SparsePauliOp(pauli_z(i)) + coeff / 2.0 * SparsePauliOp(pauli_id)
                )
            )

        z_operator = reduce(operator.add, z_summands)
        spin_op_encoding.append(z_operator)

        # 4. add the identity operator
        spin_op_encoding.append(PauliSumOp(1.0 * SparsePauliOp(pauli_id)))

        # return the lookup table for the transformed XYZI operators
        return spin_op_encoding
