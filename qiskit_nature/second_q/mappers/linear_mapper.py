# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Linear Mapper."""

from __future__ import annotations
import operator

from collections import defaultdict
from fractions import Fraction
from functools import reduce

import numpy as np

from qiskit.quantum_info import Pauli, SparsePauliOp

from qiskit_nature.second_q.operators import SpinOp
from .spin_mapper import SpinMapper


class LinearMapper(SpinMapper):
    """The Linear spin-to-qubit mapping."""

    def _map_single(
        self, second_q_op: SpinOp, *, register_length: int | None = None
    ) -> SparsePauliOp:
        if register_length is None:
            register_length = second_q_op.register_length

        qubit_ops_list: list[SparsePauliOp] = []

        # get linear encoding of the general spin matrices
        spinx, spiny, spinz, identity = self._linear_encoding(second_q_op.spin)
        ordered_op = second_q_op.index_order()

        char_map = {"X": spinx, "Y": spiny, "Z": spinz}

        for terms, coeff in ordered_op.terms():
            mat = defaultdict(list)  # type: dict[int, list]
            for op, idx in terms:
                if idx not in mat:
                    mat[idx] = identity
                mat[idx] = mat[idx] @ char_map[op]

            operatorlist = [mat[i] if i in mat else identity for i in range(register_length)]
            # Now, we can tensor all operators in this list
            qubit_ops_list.append(coeff * reduce(operator.xor, reversed(operatorlist)))

        qubit_op = reduce(operator.add, qubit_ops_list)
        return qubit_op.simplify()

    def _linear_encoding(self, spin: Fraction | float) -> list[SparsePauliOp]:
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
        for i, coeff in enumerate(np.diag(SpinOp.x(spin).to_matrix(), 1)):
            x_summands.append(
                coeff / 2.0 * SparsePauliOp(pauli_x(i).dot(pauli_x(i + 1)))
                + coeff / 2.0 * SparsePauliOp(pauli_y(i).dot(pauli_y(i + 1)))
            )

        # 2. build the non-diagonal Y operator
        y_summands = []
        for i, coeff in enumerate(np.diag(SpinOp.y(spin).to_matrix(), 1)):
            y_summands.append(
                -1j * coeff / 2.0 * SparsePauliOp(pauli_x(i).dot(pauli_y(i + 1)))
                + 1j * coeff / 2.0 * SparsePauliOp(pauli_y(i).dot(pauli_x(i + 1)))
            )

        # 3. build the diagonal Z
        z_summands = []
        for i, coeff in enumerate(np.diag(SpinOp.z(spin).to_matrix())):
            # get the first upper diagonal of coeff.
            z_summands.append(
                coeff / 2.0 * SparsePauliOp(pauli_z(i)) + coeff / 2.0 * SparsePauliOp(pauli_id)
            )

        # return the lookup table for the transformed XYZI operators
        spin_op_encoding = [
            reduce(operator.add, x_summands),
            reduce(operator.add, y_summands),
            reduce(operator.add, z_summands),
            SparsePauliOp(pauli_id),
        ]
        return spin_op_encoding
