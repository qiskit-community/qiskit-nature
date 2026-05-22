# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Bravyi-Kitaev Mapper."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from qiskit.quantum_info.operators import Pauli

from .fermionic_mapper import FermionicMapper
from .mode_based_mapper import ModeBasedMapper, PauliType


class BravyiKitaevMapper(FermionicMapper, ModeBasedMapper):
    """The Bravyi-Kitaev fermion-to-qubit mapping."""

    def pauli_table(self, register_length: int) -> list[tuple[PauliType, PauliType]]:
        return self._pauli_table(register_length)

    @staticmethod
    @lru_cache(maxsize=32)
    def _pauli_table(register_length: int) -> list[tuple[PauliType, PauliType]]:
        def parity_set(j, n):
            """
            Computes the parity set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indices
            """
            indices = np.array([])
            if n % 2 != 0:
                return indices

            if j < n / 2:
                indices = np.append(indices, parity_set(j, n / 2))
            else:
                indices = np.append(
                    indices, np.append(parity_set(j - n / 2, n / 2) + n / 2, n / 2 - 1)
                )
            return indices

        def update_set(j, n):
            """
            Computes the update set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indices
            """
            indices = np.array([])
            if n % 2 != 0:
                return indices
            if j < n / 2:
                indices = np.append(indices, np.append(n - 1, update_set(j, n / 2)))
            else:
                indices = np.append(indices, update_set(j - n / 2, n / 2) + n / 2)
            return indices

        def flip_set(j, n):
            """
            Computes the flip set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indices
            """
            indices = np.array([])
            if n % 2 != 0:
                return indices
            if j < n / 2:
                indices = np.append(indices, flip_set(j, n / 2))
            elif n / 2 <= j < n - 1:
                indices = np.append(indices, flip_set(j - n / 2, n / 2) + n / 2)
            else:
                indices = np.append(
                    np.append(indices, flip_set(j - n / 2, n / 2) + n / 2), n / 2 - 1
                )
            return indices

        pauli_table = []
        # FIND BINARY SUPERSET SIZE
        bin_sup = 1
        while register_length > np.power(2, bin_sup):
            bin_sup += 1
        # DEFINE INDEX SETS FOR EVERY FERMIONIC MODE
        update_sets = []
        update_pauli = []

        parity_sets = []
        parity_pauli = []

        flip_sets = []

        remainder_sets = []
        remainder_pauli = []
        for j in range(register_length):

            update_sets.append(update_set(j, np.power(2, bin_sup)))
            update_sets[j] = update_sets[j][update_sets[j] < register_length]

            parity_sets.append(parity_set(j, np.power(2, bin_sup)))
            parity_sets[j] = parity_sets[j][parity_sets[j] < register_length]

            flip_sets.append(flip_set(j, np.power(2, bin_sup)))
            flip_sets[j] = flip_sets[j][flip_sets[j] < register_length]

            remainder_sets.append(np.setdiff1d(parity_sets[j], flip_sets[j]))

            update_pauli.append(
                Pauli(
                    (np.zeros(register_length, dtype=bool), np.zeros(register_length, dtype=bool))
                )
            )
            parity_pauli.append(
                Pauli(
                    (np.zeros(register_length, dtype=bool), np.zeros(register_length, dtype=bool))
                )
            )
            remainder_pauli.append(
                Pauli(
                    (np.zeros(register_length, dtype=bool), np.zeros(register_length, dtype=bool))
                )
            )
            for k in range(register_length):
                if np.isin(k, update_sets[j]):
                    update_pauli[j].x[k] = True
                if np.isin(k, parity_sets[j]):
                    parity_pauli[j].z[k] = True
                if np.isin(k, remainder_sets[j]):
                    remainder_pauli[j].z[k] = True

            x_j = Pauli(
                (np.zeros(register_length, dtype=bool), np.zeros(register_length, dtype=bool))
            )
            x_j.x[j] = True
            y_j = Pauli(
                (np.zeros(register_length, dtype=bool), np.zeros(register_length, dtype=bool))
            )
            y_j.z[j] = True
            y_j.x[j] = True
            pauli_table.append(
                (
                    parity_pauli[j] & x_j & update_pauli[j],
                    remainder_pauli[j] & y_j & update_pauli[j],
                )
            )

        # PauliList has the phase information.
        # Here, phase is unnecessary, so the following removes phase.
        for pauli1, pauli2 in pauli_table:
            pauli1.phase = 0
            pauli2.phase = 0

        return pauli_table
