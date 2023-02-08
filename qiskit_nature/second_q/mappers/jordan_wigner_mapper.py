# This code is part of Qiskit.
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

"""The Jordan-Wigner Mapper. """

from __future__ import annotations

from functools import lru_cache

import numpy as np

from qiskit.quantum_info.operators import Pauli

from qiskit_nature.deprecation import deprecate_arguments
from .fermionic_mapper import FermionicMapper


class JordanWignerMapper(FermionicMapper):
    """The Jordan-Wigner fermion-to-qubit mapping."""

    @classmethod
    @deprecate_arguments("0.6.0", {"nmodes": "register_length"})
    @lru_cache(maxsize=32)
    def pauli_table(
        cls, register_length: int, *, nmodes: int | None = None
    ) -> list[tuple[Pauli, Pauli]]:
        # pylint: disable=unused-argument
        pauli_table = []

        for i in range(register_length):
            a_z = np.asarray([1] * i + [0] + [0] * (register_length - i - 1), dtype=bool)
            a_x = np.asarray([0] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            b_z = np.asarray([1] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            b_x = np.asarray([0] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            # c_z = np.asarray([0] * i + [1] + [0] * (register_length - i - 1), dtype=bool)
            # c_x = np.asarray([0] * register_length, dtype=bool)
            pauli_table.append((Pauli((a_z, a_x)), Pauli((b_z, b_x))))
            # TODO add Pauli 3-tuple to lookup table

        return pauli_table
