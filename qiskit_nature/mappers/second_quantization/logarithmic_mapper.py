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

import operator

from fractions import Fraction
from functools import reduce
from typing import List, Union

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli, SparsePauliOp, Operator
from qiskit_nature.operators.second_quantization import SpinOp
from .spin_mapper import SpinMapper


class LogarithmicMapper(SpinMapper):  # pylint: disable=missing-class-docstring
    def __init__(self, embed_padding=1, embed_location='upper'):
        """The Linear spin-to-qubit mapping."""
        super().__init__(allows_two_qubit_reduction=False)
        self._embed_padding = embed_padding
        self._embed_location = embed_location

    def map(self, second_q_op: SpinOp) -> PauliSumOp:

        qubit_ops_list: List[PauliSumOp] = []

        # get linear encoding of the general spin matrices
        spinx, spiny, spinz, identity = self._logarithmic_encoding(second_q_op.spin)
        print(spinx, spiny, spinz, identity)

        for idx, (_, coeff) in enumerate(second_q_op.to_list()):

            operatorlist: List[PauliSumOp] = []

            for n_x, n_y, n_z in zip(second_q_op.x[idx], second_q_op.y[idx], second_q_op.z[idx]):

                print(n_x, n_y, n_z)
                operator_on_spin_i: List[PauliSumOp] = []

                if n_x > 0:
                    operator_on_spin_i.append(reduce(operator.matmul, [spinx] * int(n_x)))

                print(type(spiny))
                print(type(reduce(operator.matmul, [spiny] * int(n_y))))

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

    def _logarithmic_encoding(self, spin: Union[Fraction, float]) -> List[PauliSumOp]:
        """
        Generates a 'local_encoding_transformation' of the spin S operators 'X', 'Y', 'Z' and 'identity'
        to qubit operators (linear combinations of pauli strings).
        In this 'local_encoding_transformation' each individual spin S system is represented via
        the lowest lying 2S+1 states in a qubit system with the minimal number of qubits needed to
        represent >= 2S+1 distinct states.
        Args:
            embed_padding: complex,
                The matrix element to which the diagonal matrix elements for the 2^nqubits - (2S+1) 'unphysical'
                states should be set to.
        Returns:
            self.transformed_XYZI: list,
                The 4-element list of transformed spin S 'X', 'Y', 'Z' and 'identity' operators.
                I.e.
                    self.transformed_XYZI[0] corresponds to the linear combination of pauli strings needed
                    to represent the embedded 'X' operator
        """
        print('Log encoding is calculated.')
        spin_op_encoding: List[PauliSumOp] = []
        dspin = int(2 * spin + 1)
        nqubits = int(np.ceil(np.log2(dspin)))

        # Get the spin matrices (from qutip)
        spin_matrices = [np.asarray(SpinOp(symbol, spin=spin).to_matrix()) for symbol in "XYZ"]

            # qt.jmat(self.S, symbol).data.todense()) for symbol in 'xyz']
        # Append the identity
        spin_matrices.append(np.eye(dspin))

        # Embed the spin matrices in a larger matrix of size 2**nqubits x 2**nqubits
        embed = lambda matrix: self._embed_matrix(matrix,
                                            nqubits,
                                            embed_padding=self._embed_padding,
                                            embed_location=self._embed_location)
        embedded_spin_matrices = list(map(embed, spin_matrices))

        # Generate aqua operators from these embeded spin matrices to then perform the Pauli-Scalar product
        embedded_operators = [Operator(matrix) for matrix in embedded_spin_matrices]
        # Perform the projections onto the pauli strings via the scalar product:
        for op in embedded_operators:
            op = SparsePauliOp.from_operator(op)
            op.chop()
            spin_op_encoding.append(PauliSumOp(1.0*op))
        return spin_op_encoding

    def _embed_matrix(self, matrix, nqubits, embed_padding=0., embed_location='upper'):
        """
        Embeds `matrix` into the upper/lower diagonal block of a 2^nqubits by 2^nqubits matrix and pads the
        diagonal of the upper left block matrix with the value of `embed_padding`. Whether the upper/lower
        diagonal block is used depends on `embed_location`.
        I.e. using embed_location = 'upper' returns the matrix:
            [[ matrix,    0             ],
            [   0   , embed_padding * I]]
        Using embed_location = 'lower' returns the matrix:
            [[ embed_padding * I,    0    ],
            [      0           ,  matrix ]]
        Args:
            matrix (numpy.ndarray):
                The matrix (2D-array) to embed
            nqubits (int):
                The number of qubits on which the embedded matrix should act on.
            embed_padding (float):
                The value of the diagonal elements of the upper left block of the embedded matrix.
            embed_location (str):
                Must be one of ['upper', 'lower']. This parameters sets whether the given matrix is embedded in the
                upper left hand corner or the lower right hand corner of the larger matrix.
        Returns:
            full_matrix (numpy.ndarray):
                If `matrix` is of size 2^nqubits, returns `matrix`.
                Else it returns the block matrix (I = identity)
                [[ embed_padding * I,    0    ],
                [      0           , `matrix`]]
        """
        full_dim = 1 << nqubits
        subs_dim = matrix.shape[0]

        dim_diff = full_dim - subs_dim
        if dim_diff == 0:
            return matrix

        elif dim_diff > 0:
            if embed_location == 'lower':
                full_matrix = np.zeros((full_dim, full_dim), dtype=complex)
                full_matrix[:dim_diff, :dim_diff] = np.eye(dim_diff) * embed_padding
                full_matrix[dim_diff:, dim_diff:] = matrix

            elif embed_location == 'upper':
                full_matrix = np.zeros((full_dim, full_dim), dtype=complex)
                full_matrix[:subs_dim, :subs_dim] = matrix
                full_matrix[subs_dim:, subs_dim:] = np.eye(dim_diff) * embed_padding

            else:
                raise UserWarning('embed_location must be one of ["upper","lower"]')

            return full_matrix

        else:
            raise UserWarning('The given matrix does not fit into the space spanned by {} qubits'.format(nqubits))