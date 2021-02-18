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
import numpy as np
from qiskit.quantum_info import Pauli


def bravyi_kitaev_mode(num_modes: int):
    """
    Bravyi-Kitaev mode.

    Args:
        num_modes (int): number of modes

     Returns:
         numpy.ndarray: Array of mode indexes
    """

    def parity_set(j: int, num_modes: int):
        """
        Computes the parity set of the j-th orbital in n modes.

        Args:
            j (int) : the orbital index
            num_modes (int) : the total number of modes

        Returns:
            numpy.ndarray: Array of mode indexes
        """
        indexes = np.array([])
        if num_modes % 2 != 0:
            return indexes

        if j < num_modes / 2:
            indexes = np.append(indexes, parity_set(j, num_modes / 2))
        else:
            indexes = np.append(indexes, np.append(
                parity_set(j - num_modes / 2, num_modes / 2) + num_modes / 2, num_modes / 2 - 1))
        return indexes

    def update_set(j: int, num_modes: int):
        """
        Computes the update set of the j-th orbital in n modes.

        Args:
            j (int) : the orbital index
            num_modes (int) : the total number of modes

        Returns:
            numpy.ndarray: Array of mode indexes
        """
        indexes = np.array([])
        if num_modes % 2 != 0:
            return indexes
        if j < num_modes / 2:
            indexes = np.append(indexes, np.append(
                num_modes - 1, update_set(j, num_modes / 2)))
        else:
            indexes = np.append(indexes,
                                update_set(j - num_modes / 2, num_modes / 2) + num_modes / 2)
        return indexes

    def flip_set(j: int, num_modes: int):
        """
        Computes the flip set of the j-th orbital in n modes.

        Args:
            j (int) : the orbital index
            num_modes (int) : the total number of modes

        Returns:
            numpy.ndarray: Array of mode indexes
        """
        indexes = np.array([])
        if num_modes % 2 != 0:
            return indexes
        if j < num_modes / 2:
            indexes = np.append(indexes, flip_set(j, num_modes / 2))
        elif num_modes / 2 <= j < num_modes - 1:
            indexes = np.append(indexes, flip_set(j - num_modes / 2, num_modes / 2) + num_modes / 2)
        else:
            indexes = np.append(np.append(indexes, flip_set(
                j - num_modes / 2, num_modes / 2) + num_modes / 2), num_modes / 2 - 1)
        return indexes

    a_list = []
    # FIND BINARY SUPERSET SIZE
    bin_sup = 1
    # pylint: disable=comparison-with-callable
    while num_modes > np.power(2, bin_sup):
        bin_sup += 1
    # DEFINE INDEX SETS FOR EVERY FERMIONIC MODE
    update_sets = []
    update_pauli = []

    parity_sets = []
    parity_pauli = []

    flip_sets = []

    remainder_sets = []
    remainder_pauli = []
    for j in range(num_modes):

        update_sets.append(update_set(j, np.power(2, bin_sup)))
        update_sets[j] = update_sets[j][update_sets[j] < num_modes]

        parity_sets.append(parity_set(j, np.power(2, bin_sup)))
        parity_sets[j] = parity_sets[j][parity_sets[j] < num_modes]

        flip_sets.append(flip_set(j, np.power(2, bin_sup)))
        flip_sets[j] = flip_sets[j][flip_sets[j] < num_modes]

        remainder_sets.append(np.setdiff1d(parity_sets[j], flip_sets[j]))

        update_pauli.append(Pauli(np.zeros(num_modes, dtype=bool), np.zeros(num_modes, dtype=bool)))
        parity_pauli.append(Pauli(np.zeros(num_modes, dtype=bool), np.zeros(num_modes, dtype=bool)))
        remainder_pauli.append(
            Pauli(np.zeros(num_modes, dtype=bool), np.zeros(num_modes, dtype=bool)))
        for k in range(num_modes):
            if np.in1d(k, update_sets[j]):
                update_pauli[j].update_x(True, k)
            if np.in1d(k, parity_sets[j]):
                parity_pauli[j].update_z(True, k)
            if np.in1d(k, remainder_sets[j]):
                remainder_pauli[j].update_z(True, k)

        x_j = Pauli(np.zeros(num_modes, dtype=bool), np.zeros(num_modes, dtype=bool))
        x_j.update_x(True, j)
        y_j = Pauli(np.zeros(num_modes, dtype=bool), np.zeros(num_modes, dtype=bool))
        y_j.update_z(True, j)
        y_j.update_x(True, j)
        a_list.append((update_pauli[j] * x_j * parity_pauli[j],
                       update_pauli[j] * y_j * remainder_pauli[j]))
    return a_list
