# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Linear Mapper for Bosons."""

from __future__ import annotations
import operator

from functools import reduce

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli, SparsePauliOp

from qiskit_nature.second_q.operators import BosonicOp
from .bosonic_mapper import BosonicMapper


class BosonicLinearMapper(BosonicMapper):
    """
    The Linear boson-to-qubit mapping.
    This mapper implements the formula in Section II.C of: https://arxiv.org/abs/2108.04258

    b_k^\\dagger -> sum(sqrt(n_k + 1) \\sigma_(n_k)^+\\sigma_(n_k + 1)^-, from n_k = 0 to n_k^max + 1)
    where n_k^max is the truncation order (defined by the user)
    In the following, we explicit the operators \\sigma^+ and \\sigma^- with the Pauli matrices:
    \\sigma_(n_k)^+ := S_j^+ = 0.5 * (X_j + i*Y_j) ; \\sigma_(n_k)^- := S_j^- = 0.5 * (X_j - i*Y_j)

    The length of the qubit register is: BosonicOp.num_mode * (BosonicLinearMapper.truncation + 1)

    e.g. let's consider a 2 mode system with truncation = 2: +_0 -_1
    Let's map each term:
    +_0 -> S_0^+ S_1^- + sqrt(2) S_1^+ S_2^- =
        (0.5)^4 * (X_0 + i*Y_0) * (X_0 - i*Y_0) * (X_1 - i*Y_1) * (X_1 + i*Y_1) +
        (0.5)^4 * (X_1 + i*Y_1) * (X_1 - i*Y_1) * (X_2 - i*Y_2) * (X_2 + i*Y_2) * sqrt(2) =
        (0.5)^4 * (XX - i*XY + i*YX + YY)IIII * I(XX + i*XY - i*YX + YY)III +
        (0.5)^4 * I(XX + i*XY - i*YX + YY)III * II(XX - i*XY + i*YX + YY)II * sqrt(2)
    -_1 -> S_0^- S_1^+ + sqrt(2) S_1^- S_2^+

    Generates a 'linear_encoding' of the Bosonic operator b_k^\\dagger, b_k to qubit operators
        (linear combinations of pauli strings).
        In this 'linear_encoding' each bosonic mode is represented via n_k^max + 1 qubits, where n_k^max
        is the truncation of the mode (meaning the number of states used in the expansion of the mode,
        or equivalently the state at which the maximum excitation can take place).
        The mode |k> is then mapped to the occupation number vector
        |0_{nk_^max}, 0_{nk_^max - 1}, ..., 0_{n_k + 1}, 1_{n_k}, 0_{n_k - 1}, ..., 0_{0_k}>
    """

    def _map_single(
        self, second_q_op: BosonicOp, *, register_length: int | None = None
    ) -> SparsePauliOp | PauliSumOp:
        # If register_length was passed, override the one present in the BosonicOp
        if register_length is None:
            register_length = second_q_op.num_modes

        qubit_register_length = register_length * (self.truncation + 1)
        # Create a Pauli operator, which we will fill in this method
        # TODO: Check if we need to normal/index_order
        pauli_op: list[SparsePauliOp] = []
        # Then we loop over all the terms of the bosonic operator
        for terms, coeff in second_q_op.terms():
            # Then loop over each term (terms -> List[Tuple[string, int]])
            bos_op_to_pauli_op = SparsePauliOp(["I" * qubit_register_length], coeffs=[1.])
            for op, idx in terms:
                if op not in ('+', '-'):
                    break
                pauli_expansion: list[SparsePauliOp] = []
                # Now we are dealing with a single bosonic operator. We have to perform the linear mapper
                for n_k in range(self.truncation):
                    prefactor = np.sqrt(n_k + 1) / 4
                    # Define the actual index in the qubit register. It is given n_k plus the shift due
                    # to the mode onto which the operator is acting
                    register_index = n_k + idx * (self.truncation + 1)
                    # Now build the Pauli operators XX, XY, YX, YY, which arise from S_i^+ S_j^-
                    x_x, x_y, y_x, y_y = self.get_ij_pauli_matrix(register_index, qubit_register_length)

                    tmp_op = SparsePauliOp(x_x) + SparsePauliOp(y_y)
                    if op == "+":
                        tmp_op += - 1j*SparsePauliOp(x_y) + 1j*SparsePauliOp(y_x)
                    else:
                        tmp_op += + 1j*SparsePauliOp(x_y) - 1j*SparsePauliOp(y_x)
                    pauli_expansion.append(prefactor * tmp_op)
                # Add the Pauli expansion for a single n_k to map of the bosonic operator
                bos_op_to_pauli_op = reduce(operator.add, pauli_expansion).compose(bos_op_to_pauli_op)
            # Add the map of the single boson op (e.g. +_0) to the map of the full bosonic operator
            pauli_op.append(coeff * reduce(operator.add, bos_op_to_pauli_op.simplify()))

        # return the lookup table for the transformed XYZI operators
        bos_op_encoding = reduce(operator.add, pauli_op)
        return bos_op_encoding

    def get_ij_pauli_matrix(self, register_index: int, register_length: int):
        "This method builds the Qiskit Pauli operators of the operators XX, YY, XY and YX"
        # Define recurrent variables
        prefix_zeros = [0] * register_index
        suffix_zeros = [0] * (register_length - 2 - register_index)
        # Build the Pauli strings
        x_x = Pauli((
            [0] * register_length,
            prefix_zeros + [1, 1] + suffix_zeros,
        ))
        x_y = Pauli((
            prefix_zeros + [1, 0] + suffix_zeros,
            prefix_zeros + [1, 1] + suffix_zeros,
        ))
        y_x = Pauli((
            prefix_zeros + [0, 1] + suffix_zeros,
            prefix_zeros + [1, 1] + suffix_zeros,
        ))
        y_y = Pauli((
            prefix_zeros + [1, 1] + suffix_zeros,
            prefix_zeros + [1, 1] + suffix_zeros
        ))
        return x_x, x_y, y_x, y_y
