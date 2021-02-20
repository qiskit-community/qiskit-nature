# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Bravyi-Kitaev Mapping interface."""

import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli

from qiskit_nature.operators.second_quantization.particle_op import ParticleOp
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp

from .qubit_mapping import QubitMapping


class BravyiKitaevMapping(QubitMapping):
    """The Bravyi-Kitaev fermion-to-qubit mapping. """

    def supports_particle_type(self, particle_type: ParticleOp) -> bool:
        """Returns whether the queried particle-type operator is supported by this mapping.

        Args:
            particle_type: the particle-type to query support for.

        Returns:
            A boolean indicating whether the queried particle-type is supported.
        """
        return isinstance(particle_type, FermionicOp)

    def map(self, second_q_op: ParticleOp) -> PauliSumOp:
        """Maps a `ParticleOp` to a `PauliSumOp` using the Bravyi-Kitaev
        fermion-to-qubit mapping.

        Args:
            second_q_op: the `ParticleOp` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.

        Raises:
            QiskitNatureError: FermionicOp has a invalid label.
            TypeError: Type of second_q_op is not FermionicOp.
        """
        if not isinstance(second_q_op, FermionicOp):
            raise TypeError(
                f"Bravyi-Kitaev mapper only maps from FermionicOp, not {type(second_q_op)}"
            )

        # number of modes/sites for the Parity transform (= number of fermionc modes)
        nmodes = second_q_op.register_length

        def parity_set(j, n):
            """
            Computes the parity set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes
            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes

            if j < n / 2:
                indexes = np.append(indexes, parity_set(j, n / 2))
            else:
                indexes = np.append(indexes, np.append(
                    parity_set(j - n / 2, n / 2) + n / 2, n / 2 - 1))
            return indexes

        def update_set(j, n):
            """
            Computes the update set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes
            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes
            if j < n / 2:
                indexes = np.append(indexes, np.append(
                    n - 1, update_set(j, n / 2)))
            else:
                indexes = np.append(indexes, update_set(j - n / 2, n / 2) + n / 2)
            return indexes

        def flip_set(j, n):
            """
            Computes the flip set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes
            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes
            if j < n / 2:
                indexes = np.append(indexes, flip_set(j, n / 2))
            elif n / 2 <= j < n - 1:
                indexes = np.append(indexes, flip_set(j - n / 2, n / 2) + n / 2)
            else:
                indexes = np.append(np.append(indexes, flip_set(
                    j - n / 2, n / 2) + n / 2), n / 2 - 1)
            return indexes

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
            pauli_table.append((update_pauli[j] * x_j * parity_pauli[j],
                                update_pauli[j] * y_j * remainder_pauli[j]))

        return QubitMapping.mode_based_mapping(second_q_op, pauli_table)
