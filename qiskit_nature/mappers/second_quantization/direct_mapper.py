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

"""The Direct Mapper."""

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli

from qiskit_nature.operators.second_quantization import VibrationOp

from .qubit_mapper import QubitMapper
from .vibration_mapper import VibrationMapper


class DirectMapper(VibrationMapper):
    """The Direct boson-to-qubit mapping. """

    def map(self, second_q_op: VibrationOp) -> PauliSumOp:

        if isinstance(second_q_op.num_modals, int):
            modals_per_mode = [second_q_op.num_modals] * second_q_op.num_modes
        else:
            modals_per_mode = second_q_op.num_modals

        pauli_tables = []
        for nmodes in modals_per_mode:
            pauli_table = []
            for i in range(nmodes):
                a_z = np.asarray([0] * i + [0] + [0] * (nmodes - i - 1), dtype=bool)
                a_x = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
                b_z = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
                b_x = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
                pauli_table.append((Pauli((a_z, a_x)), Pauli((b_z, b_x))))
            pauli_tables.append(pauli_table)

        return QubitMapper.mode_based_mapping(second_q_op, pauli_tables)
