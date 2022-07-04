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

"""The Direct Mapper."""

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli

from qiskit_nature.second_q.operators import VibrationalOp
from .qubit_mapper import QubitMapper


class DirectMapper(QubitMapper):  # pylint: disable=missing-class-docstring
    def __init__(self):
        """The Direct mapper.

        This mapper maps a :class:`~.VibrationalOp` to a :class:`PauliSumOp`.
        In doing so, each modal of the the `VibrationalOp` gets mapped to a single qubit.
        """
        super().__init__(allows_two_qubit_reduction=False)

    def map(self, second_q_op: VibrationalOp) -> PauliSumOp:

        nmodes = sum(second_q_op.num_modals)

        pauli_table = []
        for i in range(nmodes):
            a_z = np.asarray([0] * i + [0] + [0] * (nmodes - i - 1), dtype=bool)
            a_x = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
            b_z = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
            b_x = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
            pauli_table.append((Pauli((a_z, a_x)), Pauli((b_z, b_x))))

        return QubitMapper.mode_based_mapping(second_q_op, pauli_table)
