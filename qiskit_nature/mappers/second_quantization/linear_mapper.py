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
