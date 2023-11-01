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

"""The Direct Mapper."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from qiskit.quantum_info.operators import Pauli

from .vibrational_mapper import VibrationalMapper


class DirectMapper(VibrationalMapper):
    """The Direct mapper.

    This mapper maps a :class:`~.VibrationalOp` to a qubit operator. In doing so, each modal of the
    ``VibrationalOp`` gets mapped to a single qubit.
    """

    @classmethod
    @lru_cache(maxsize=32)
    def pauli_table(cls, register_length: int) -> list[tuple[Pauli, Pauli]]:
        # pylint: disable=unused-argument
        pauli_table = []

        for i in range(register_length):
            a_z = np.asarray([0] * i + [0] + [0] * (register_length - i - 1), dtype=bool)
            a_x = np.asarray([0] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            b_z = np.asarray([0] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            b_x = np.asarray([0] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            pauli_table.append((Pauli((a_z, a_x)), Pauli((b_z, b_x))))

        return pauli_table
