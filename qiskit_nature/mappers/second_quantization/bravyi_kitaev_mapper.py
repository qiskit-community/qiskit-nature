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

"""The Bravyi-Kitaev Mapper."""

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli

from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp

from .fermionic_mapper import FermionicMapper
from .qubit_mapper import QubitMapper


class BravyiKitaevMapper(FermionicMapper):
    """The Bravyi-Kitaev fermion-to-qubit mapping. """

    def map(self, second_q_op: FermionicOp) -> PauliSumOp:

        # number of modes/sites for the BK transform (= number of fermionic modes)
        nmodes = second_q_op.register_length

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
                indices = np.append(indices, np.append(
                    parity_set(j - n / 2, n / 2) + n / 2, n / 2 - 1))
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
                indices = np.append(indices, np.append(
                    n - 1, update_set(j, n / 2)))
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
                indices = np.append(np.append(indices, flip_set(
                    j - n / 2, n / 2) + n / 2), n / 2 - 1)
            return indices

        pauli_table = []
        # FIND BINARY SUPERSET SIZE
        bin_sup = 1
        # pylint: disable=comparison-with-callable
        while nmodes > np.power(2, bin_sup):
            bin_sup += 1
        # DEFINE INDEX SETS FOR EVERY FERMIONIC MODE
        update_sets = []
        update_pauli = []

        parity_sets = []
        parity_pauli = []

        flip_sets = []

        remainder_sets = []
        remainder_pauli = []
        for j in range(nmodes):

            update_sets.append(update_set(j, np.power(2, bin_sup)))
            update_sets[j] = update_sets[j][update_sets[j] < nmodes]

            parity_sets.append(parity_set(j, np.power(2, bin_sup)))
            parity_sets[j] = parity_sets[j][parity_sets[j] < nmodes]

            flip_sets.append(flip_set(j, np.power(2, bin_sup)))
            flip_sets[j] = flip_sets[j][flip_sets[j] < nmodes]

            remainder_sets.append(np.setdiff1d(parity_sets[j], flip_sets[j]))

            update_pauli.append(Pauli((np.zeros(nmodes, dtype=bool),
                                       np.zeros(nmodes, dtype=bool))))
            parity_pauli.append(Pauli((np.zeros(nmodes, dtype=bool),
                                       np.zeros(nmodes, dtype=bool))))
            remainder_pauli.append(Pauli((np.zeros(nmodes, dtype=bool),
                                          np.zeros(nmodes, dtype=bool))))
            for k in range(nmodes):
                if np.in1d(k, update_sets[j]):
                    update_pauli[j].x[k] = True
                if np.in1d(k, parity_sets[j]):
                    parity_pauli[j].z[k] = True
                if np.in1d(k, remainder_sets[j]):
                    remainder_pauli[j].z[k] = True

            x_j = Pauli((np.zeros(nmodes, dtype=bool), np.zeros(nmodes, dtype=bool)))
            x_j.x[j] = True
            y_j = Pauli((np.zeros(nmodes, dtype=bool), np.zeros(nmodes, dtype=bool)))
            y_j.z[j] = True
            y_j.x[j] = True
            pauli_table.append((parity_pauli[j] & x_j & update_pauli[j],
                                remainder_pauli[j] & y_j & update_pauli[j]))

        return QubitMapper.mode_based_mapping(second_q_op, pauli_table)
