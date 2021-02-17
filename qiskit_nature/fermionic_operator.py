# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Fermionic Operator """

import itertools
import logging

import numpy as np

from .particle_hole import particle_hole_transformation

logger = logging.getLogger(__name__)


class FermionicOperator:
    """
    A set of functions to map fermionic Hamiltonians to qubit Hamiltonians.

    References:

    - *E. Wigner and P. Jordan., Über das Paulische Äquivalenzverbot,
      Z. Phys., 47:631 (1928).*
    - *S. Bravyi and A. Kitaev. Fermionic quantum computation,
      Ann. of Phys., 298(1):210–226 (2002).*
    - *A. Tranter, S. Sofia, J. Seeley, M. Kaicher, J. McClean, R. Babbush,
      P. Coveney, F. Mintert, F. Wilhelm, and P. Love. The Bravyi–Kitaev
      transformation: Properties and applications. Int. Journal of Quantum
      Chemistry, 115(19):1431–1441 (2015).*
    - *S. Bravyi, J. M. Gambetta, A. Mezzacapo, and K. Temme,
      arXiv e-print arXiv:1701.08213 (2017).*
    - *K. Setia, J. D. Whitfield, arXiv:1712.00446 (2017)*

    """

    def __init__(self, h1, h2=None, ph_trans_shift=None):
        """
        This class requires the integrals stored in the '*chemist*' notation

             h2(i,j,k,l) --> adag_i adag_k a_l a_j

        and the integral values are used for the coefficients of the second-quantized
        Hamiltonian that is built. The integrals input here should be in block spin
        format and also have indexes reordered as follows 'ijkl->ljik'

        There is another popular notation, the '*physicist*' notation

             h2(i,j,k,l) --> adag_i adag_j a_k a_l

        If you are using the '*physicist*' notation, you need to convert it to
        the '*chemist*' notation. E.g. h2=numpy.einsum('ikmj->ijkm', h2)

        The :class:`~qiskit_nature.drivers.QMolecule` class has
        :attr:`~qiskit_nature.drivers.QMolecule.one_body_integrals` and
        :attr:`~qiskit_nature.drivers.QMolecule.two_body_integrals` properties that can be
        directly supplied to the `h1` and `h2` parameters here respectively.

        Args:
            h1 (numpy.ndarray): second-quantized fermionic one-body operator, a 2-D (NxN) tensor
            h2 (numpy.ndarray): second-quantized fermionic two-body operator,
                                a 4-D (NxNxNxN) tensor
            ph_trans_shift (float): energy shift caused by particle hole transformation
        """
        self._h1 = h1
        if h2 is None:
            h2 = np.zeros((h1.shape[0], h1.shape[0], h1.shape[0], h1.shape[0]), dtype=h1.dtype)
        self._h2 = h2
        self._ph_trans_shift = ph_trans_shift
        self._modes = self._h1.shape[0]
        self._map_type = None

    @property
    def modes(self):
        """Getter of modes."""
        return self._modes

    @property
    def h1(self):  # pylint: disable=invalid-name
        """Getter of one body integral tensor."""
        return self._h1

    @h1.setter
    def h1(self, new_h1):  # pylint: disable=invalid-name
        """Setter of one body integral tensor."""
        self._h1 = new_h1

    @property
    def h2(self):  # pylint: disable=invalid-name
        """Getter of two body integral tensor."""
        return self._h2

    @h2.setter
    def h2(self, new_h2):  # pylint: disable=invalid-name
        """Setter of two body integral tensor."""
        self._h2 = new_h2

    def __eq__(self, other):
        """Overload == ."""
        ret = np.all(self._h1 == other._h1)
        if not ret:
            return ret
        ret = np.all(self._h2 == other._h2)
        return ret

    def __ne__(self, other):
        """Overload != ."""
        return not self.__eq__(other)

    def transform(self, unitary_matrix):
        """Transform the one and two body term based on unitary_matrix."""
        self._h1_transform(unitary_matrix)
        self._h2_transform(unitary_matrix)

    def _h1_transform(self, unitary_matrix):
        """
        Transform h1 based on unitary matrix, and overwrite original property.

        Args:
            unitary_matrix (numpy.ndarray): A 2-D unitary matrix for h1 transformation.
        """
        self._h1 = unitary_matrix.T.conj().dot(self._h1.dot(unitary_matrix))

    def _h2_transform(self, unitary_matrix):
        """
        Transform h2 to get fermionic hamiltonian, and overwrite original property.

        Args:
            unitary_matrix (numpy.ndarray): A 2-D unitary matrix for h1 transformation.
        """
        num_modes = unitary_matrix.shape[0]
        temp_ret = np.zeros((num_modes, num_modes, num_modes, num_modes),
                            dtype=unitary_matrix.dtype)
        unitary_matrix_dagger = np.conjugate(unitary_matrix)

        # option 3: temp1 is a 3-D tensor, temp2 and temp3 are 2-D tensors
        # pylint: disable=unsubscriptable-object
        for a_i in range(num_modes):
            temp1 = np.einsum('i,i...->...', unitary_matrix_dagger[:, a_i], self._h2)
            for b_i in range(num_modes):
                temp2 = np.einsum('j,j...->...', unitary_matrix[:, b_i], temp1)
                temp3 = np.einsum('kc,k...->...c', unitary_matrix_dagger, temp2)
                temp_ret[a_i, b_i, :, :] = np.einsum('ld,l...c->...cd', unitary_matrix, temp3)

        self._h2 = temp_ret

    def _convert_to_interleaved_spins(self):
        """
        Converting the spin order from block to interleaved.

        From up-up-up-up-down-down-down-down
        to up-down-up-down-up-down-up-down
        """
        # pylint: disable=unsubscriptable-object
        matrix = np.zeros((self._h1.shape), self._h1.dtype)
        n = matrix.shape[0]
        j = np.arange(n // 2)
        matrix[j, 2 * j] = 1.0
        matrix[j + n // 2, 2 * j + 1] = 1.0
        self.transform(matrix)

    def _convert_to_block_spins(self):
        """
        Converting the spin order from interleaved to block.

        From up-down-up-down-up-down-up-down
        to up-up-up-up-down-down-down-down
        """
        # pylint: disable=unsubscriptable-object
        matrix = np.zeros((self._h1.shape), self._h1.dtype)
        n = matrix.shape[0]
        j = np.arange(n // 2)
        matrix[2 * j, j] = 1.0
        matrix[2 * j + 1, n // 2 + j] = 1.0
        self.transform(matrix)

    # Modified for Open-Shell : 17.07.2019 by iso and bpa
    def particle_hole_transformation(self, num_particles):
        """
        The 'standard' second quantized Hamiltonian can be transformed in the
        particle-hole (p/h) picture, which makes the expansion of the trail wavefunction
        from the HF reference state more natural. In fact, for both trail wavefunctions
        implemented in q-lib ('heuristic' hardware efficient and UCCSD) the p/h Hamiltonian
        improves the speed of convergence of the VQE algorithm for the calculation of
        the electronic ground state properties.
        For more information on the p/h formalism see:
        P. Barkoutsos, arXiv:1805.04340(https://arxiv.org/abs/1805.04340).

        Args:
            num_particles (list, int): number of particles, if it is a list,
                                       the first number is alpha
                                       and the second number is beta.
        Returns:
            tuple: new_fer_op, energy_shift
        """

        self._convert_to_interleaved_spins()
        h_1, h_2, energy_shift = particle_hole_transformation(self._modes, num_particles,
                                                              self._h1, self._h2)
        new_fer_op = FermionicOperator(h1=h_1, h2=h_2, ph_trans_shift=energy_shift)
        new_fer_op._convert_to_block_spins()
        return new_fer_op, energy_shift

    def fermion_mode_elimination(self, fermion_mode_array):
        """Eliminate modes.

        Generate a new fermionic operator with the modes in fermion_mode_array deleted

        Args:
            fermion_mode_array (list): orbital index for elimination

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        fermion_mode_array = np.sort(fermion_mode_array)
        n_modes_old = self._modes
        n_modes_new = n_modes_old - fermion_mode_array.size
        mode_set_diff = np.setdiff1d(np.arange(n_modes_old), fermion_mode_array)
        h1_id_i, h1_id_j = np.meshgrid(mode_set_diff, mode_set_diff, indexing='ij')
        h1_new = self._h1[h1_id_i, h1_id_j].copy()
        if np.count_nonzero(self._h2) > 0:
            h2_id_i, h2_id_j, h2_id_k, h2_id_l = np.meshgrid(
                mode_set_diff, mode_set_diff, mode_set_diff, mode_set_diff, indexing='ij')
            h2_new = self._h2[h2_id_i, h2_id_j, h2_id_k, h2_id_l].copy()
        else:
            h2_new = np.zeros((n_modes_new, n_modes_new, n_modes_new, n_modes_new))
        return FermionicOperator(h1_new, h2_new)

    def fermion_mode_freezing(self, fermion_mode_array):
        """
        Freezing modes and extracting its energy.

        Generate a fermionic operator with the modes in fermion_mode_array deleted and
        provide the shifted energy after freezing.

        Args:
            fermion_mode_array (list): orbital index for freezing

        Returns:
            tuple(FermionicOperator, float):
                Fermionic Hamiltonian and energy of frozen modes
        """
        fermion_mode_array = np.sort(fermion_mode_array)
        n_modes_old = self._modes
        n_modes_new = n_modes_old - fermion_mode_array.size
        mode_set_diff = np.setdiff1d(np.arange(n_modes_old), fermion_mode_array)

        h_1 = self._h1.copy()
        h2_new = np.zeros((n_modes_new, n_modes_new, n_modes_new, n_modes_new))

        energy_shift = 0.0
        if np.count_nonzero(self._h2) > 0:
            # First simplify h2 and renormalize original h1
            for __i, __j, __l, __k in itertools.product(range(n_modes_old), repeat=4):
                # Untouched terms
                h2_ijlk = self._h2[__i, __j, __l, __k]
                if h2_ijlk == 0.0:
                    continue
                if __i in mode_set_diff and __j in mode_set_diff \
                        and __l in mode_set_diff and __k in mode_set_diff:
                    h2_new[__i - np.where(fermion_mode_array < __i)[0].size,
                           __j - np.where(fermion_mode_array < __j)[0].size,
                           __l - np.where(fermion_mode_array < __l)[0].size,
                           __k - np.where(fermion_mode_array < __k)[0].size] = h2_ijlk
                else:
                    if __i in fermion_mode_array:
                        if __l not in fermion_mode_array:
                            if __i == __k and __j not in fermion_mode_array:
                                h_1[__l, __j] -= h2_ijlk
                            elif __i == __j and __k not in fermion_mode_array:
                                h_1[__l, __k] += h2_ijlk
                        elif __i != __l:
                            if __j in fermion_mode_array and __i == __k and __l == __j:
                                energy_shift -= h2_ijlk
                            elif __l in fermion_mode_array and __i == __j and __l == __k:
                                energy_shift += h2_ijlk
                    elif __i not in fermion_mode_array and __l in fermion_mode_array:
                        if __l == __k and __j not in fermion_mode_array:
                            h_1[__i, __j] += h2_ijlk
                        elif __l == __j and __k not in fermion_mode_array:
                            h_1[__i, __k] -= h2_ijlk

        # now simplify h1
        energy_shift += np.sum(np.diagonal(h_1)[fermion_mode_array])
        h1_id_i, h1_id_j = np.meshgrid(mode_set_diff, mode_set_diff, indexing='ij')
        h1_new = h_1[h1_id_i, h1_id_j]

        return FermionicOperator(h1_new, h2_new), energy_shift

    def total_particle_number(self):
        """
        A data_preprocess_helper fermionic operator which can be used to evaluate the number of
        particle of the given eigenstate.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        modes = self._modes
        h_1 = np.eye(modes, dtype=complex)
        h_2 = np.zeros((modes, modes, modes, modes))
        return FermionicOperator(h_1, h_2)

    def total_magnetization(self):
        """
        A data_preprocess_helper fermionic operator which can be used to
        evaluate the magnetization of the given eigenstate.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        modes = self._modes
        h_1 = np.eye(modes, dtype=complex) * 0.5
        h_1[modes // 2:, modes // 2:] *= -1.0
        h_2 = np.zeros((modes, modes, modes, modes))
        return FermionicOperator(h_1, h_2)

    def _s_x_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        num_modes = self._modes
        num_modes_2 = num_modes // 2
        h_1 = np.zeros((num_modes, num_modes))
        h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
            if p != q:
                h_2[p, p + num_modes_2, q, q + num_modes_2] += 1.0
                h_2[p + num_modes_2, p, q, q + num_modes_2] += 1.0
                h_2[p, p + num_modes_2, q + num_modes_2, q] += 1.0
                h_2[p + num_modes_2, p, q + num_modes_2, q] += 1.0
            else:
                h_2[p, p + num_modes_2, p, p + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p, p + num_modes_2, p] -= 1.0
                h_2[p, p, p + num_modes_2, p + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p + num_modes_2, p, p] -= 1.0

                h_1[p, p] += 1.0
                h_1[p + num_modes_2, p + num_modes_2] += 1.0

        h_1 *= 0.25
        h_2 *= 0.25
        return h_1, h_2

    def _s_y_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        num_modes = self._modes
        num_modes_2 = num_modes // 2
        h_1 = np.zeros((num_modes, num_modes))
        h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
            if p != q:
                h_2[p, p + num_modes_2, q, q + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p, q, q + num_modes_2] += 1.0
                h_2[p, p + num_modes_2, q + num_modes_2, q] += 1.0
                h_2[p + num_modes_2, p, q + num_modes_2, q] -= 1.0
            else:
                h_2[p, p + num_modes_2, p, p + num_modes_2] += 1.0
                h_2[p + num_modes_2, p, p + num_modes_2, p] += 1.0
                h_2[p, p, p + num_modes_2, p + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p + num_modes_2, p, p] -= 1.0

                h_1[p, p] += 1.0
                h_1[p + num_modes_2, p + num_modes_2] += 1.0

        h_1 *= 0.25
        h_2 *= 0.25
        return h_1, h_2

    def _s_z_squared(self):
        """

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        num_modes = self._modes
        num_modes_2 = num_modes // 2
        h_1 = np.zeros((num_modes, num_modes))
        h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

        for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
            if p != q:
                h_2[p, p, q, q] += 1.0
                h_2[p + num_modes_2, p + num_modes_2, q, q] -= 1.0
                h_2[p, p, q + num_modes_2, q + num_modes_2] -= 1.0
                h_2[p + num_modes_2, p + num_modes_2,
                    q + num_modes_2, q + num_modes_2] += 1.0
            else:
                h_2[p, p + num_modes_2, p + num_modes_2, p] += 1.0
                h_2[p + num_modes_2, p, p, p + num_modes_2] += 1.0

                h_1[p, p] += 1.0
                h_1[p + num_modes_2, p + num_modes_2] += 1.0

        h_1 *= 0.25
        h_2 *= 0.25
        return h_1, h_2

    def total_angular_momentum(self):
        """
        Total angular momentum.

        A data_preprocess_helper fermionic operator which can be used to evaluate the total
        angular momentum of the given eigenstate.

        Returns:
            FermionicOperator: Fermionic Hamiltonian
        """
        x_h1, x_h2 = self._s_x_squared()
        y_h1, y_h2 = self._s_y_squared()
        z_h1, z_h2 = self._s_z_squared()
        h_1 = x_h1 + y_h1 + z_h1
        h_2 = x_h2 + y_h2 + z_h2

        return FermionicOperator(h1=h_1, h2=h_2)
