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

from collections import defaultdict
from fractions import Fraction
from functools import reduce

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli, SparsePauliOp

from qiskit_nature import settings
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
        if register_length is None:
            register_length = second_q_op.num_modes * (self.truncation + 1)

        # Create a Pauli operator, which we will fill in this method
        # TODO: Check if we need to normal/index_order
        pauli_op = []  # type: List[SparsePauliOp]
        # Then we loop over all the terms of the bosonic operator
        for terms, coeff in second_q_op.terms():
            # Then loop over each term (terms -> List[Tuple[string, int]])
            boson_op_to_pauli_op = []  # type: List[SparsePauliOp]
            for op, idx in terms:
                if op != "+" and op != "-":
                    break
                # Now we dealing with a single bosonic operator. We have to perform the linear mapper
                for n_k in range(self.truncation):
                    prefactor = np.sqrt(n_k + 1) / 4
                    # Define the actual index in the qubit register. It is given n_k plus the shift due
                    # to the mode onto which the operator is acting
                    register_index = n_k + idx * (self.truncation + 1)
                    # Now build the Pauli operators XX, XY, YX, YY, which arise from S_j^+ S_j^-
                    xx, xy, yx, yy = self.get_ij_pauli_matrix(register_index, register_length)

                    tmp_op = (SparsePauliOp(xx) + SparsePauliOp(yy)
                              - 1j*SparsePauliOp(xy) + 1j*SparsePauliOp(yx))
                    if op == "-":
                        tmp_op = tmp_op.conjugate()
                    # Add the Pauli expansion for a single n_k to map of the bosonic operator
                    boson_op_to_pauli_op.append(prefactor * tmp_op)
                # Add the map of the single boson op (e.g. +_0) to the map of the full bosonic operator
                pauli_op.append(coeff * reduce(operator.add, boson_op_to_pauli_op))

        # return the lookup table for the transformed XYZI operators
        bos_op_encoding = reduce(operator.add, pauli_op)
        return bos_op_encoding

    def get_ij_pauli_matrix(self, register_index: int, register_length: int):
        xx = Pauli(
            [0] * register_length,
            [0] * register_index + [1, 1] + [0] * (register_length - 2 - register_index),
        )
        xy = Pauli(
            [0] * register_index + [0, 1] + [0] * (register_length - 2 - register_index),
            [0] * register_index + [1, 1] + [0] * (register_length - 2 - register_index),
        )
        yx = Pauli(
            [0] * register_index + [1, 0] + [0] * (register_length - 2 - register_index),
            [0] * register_index + [1, 1] + [0] * (register_length - 2 - register_index),
        )
        yy = Pauli(
            [0] * register_index + [1, 1] + [0] * (register_length - 2 - register_index),
            [0] * register_index + [1, 1] + [0] * (register_length - 2 - register_index)
        )
        return xx, xy, yx, yy
