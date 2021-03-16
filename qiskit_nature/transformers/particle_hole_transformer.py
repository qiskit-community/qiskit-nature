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

"""The Particle/Hole Transformer interface."""

from typing import Optional
import copy
import numpy as np

from .. import QiskitNatureError

from .base_transformer import BaseTransformer
from ..drivers import QMolecule


class ParticleHoleTransformer(BaseTransformer):
    """The Particle/Hole transformer."""

    def __init__(self, num_electrons: int, num_orbitals: int, num_alpha: int):
        """Initializes a transformer which can reduce a `QMolecule` to a configured active space.

        Args:
            num_electrons: the number of electrons.
            num_orbitals: the number of orbitals.
            num_alpha: the number of alpha electrons.
        """
        self.num_electrons = num_electrons
        self.num_orbitals = num_orbitals
        self.num_alpha = num_alpha
        self._h1: Optional[np.ndarray] = None
        self._h2: Optional[np.ndarray] = None

    def transform(self, q_molecule: QMolecule) -> QMolecule:
        """Transforms the given `QMolecule` into the particle/hole view.

        Args:
            q_molecule: the `QMolecule` to be transformed.

        Raises:
            QiskitNatureError: Particle Hole Transformer does not work with UHF

        Returns:
            A new `QMolecule` instance.
        """

        n_qubits = self.num_orbitals * 2

        self._h1 = q_molecule.one_body_integrals
        self._h2 = q_molecule.two_body_integrals

        self._convert_to_interleaved_spins()

        h1_old_matrix = self._h1
        h2_old_matrix = self._h2

        num_alpha = self.num_alpha
        num_beta = self.num_electrons - self.num_alpha

        h1_new_sum = np.zeros([n_qubits, n_qubits])
        h2_new_sum = np.zeros([n_qubits, n_qubits, n_qubits, n_qubits])

        h2_old_matrix = -2 * h2_old_matrix.copy()
        h2_old_matrix = np.einsum('IJKL->IKLJ', h2_old_matrix.copy())

        h1_old_matrix = h1_old_matrix.copy()

        # put labels of occupied orbitals in the list in interleaved spin convention
        n_occupied = []
        for a_i in range(num_alpha):
            n_occupied.append(2 * a_i)
        for b in range(num_beta):
            n_occupied.append(2 * b + 1)

        for r in range(n_qubits):
            for s in range(n_qubits):  # pylint: disable=invalid-name
                for i in n_occupied:
                    h1_old_matrix[r][s] += h2_old_matrix[r][i][s][i].copy() - \
                                           h2_old_matrix[r][i][i][s].copy()
        identities_new_sum = 0

        for i in range(n_qubits):
            for j in range(n_qubits):
                indices_1 = [i, j]
                array_mapping_1 = ['adag', 'a']

                h1_new_matrix = np.zeros([n_qubits, n_qubits])
                h2_new_matrix = np.zeros([n_qubits, n_qubits, n_qubits, n_qubits])

                h1_new_matrix, h2_new_matrix, identities = self._normal_order_integrals(
                    n_qubits, n_occupied, indices_1, array_mapping_1, h1_old_matrix,
                    h2_old_matrix, h1_new_matrix, h2_new_matrix)

                h1_new_sum += h1_new_matrix
                h2_new_sum += h2_new_matrix
                identities_new_sum += identities

        for i in range(n_qubits):
            for j in range(n_qubits):
                for k in range(n_qubits):
                    for l_i in range(n_qubits):
                        array_to_be_ordered = [i, j, k, l_i]

                        array_mapping_2 = ['adag', 'adag', 'a', 'a']

                        h1_new_matrix = np.zeros([n_qubits, n_qubits])
                        h2_new_matrix = np.zeros([n_qubits, n_qubits, n_qubits, n_qubits])

                        h1_new_matrix, h2_new_matrix, identities = self._normal_order_integrals(
                            n_qubits, n_occupied, array_to_be_ordered, array_mapping_2,
                            h1_old_matrix, h2_old_matrix, h1_new_matrix, h2_new_matrix)

                        h1_new_sum += h1_new_matrix
                        h2_new_sum += h2_new_matrix
                        identities_new_sum += identities

        h2_new_sum = np.einsum('IKMJ->IJKM', h2_new_sum)

        self._h1 = h1_new_sum
        self._h2 = h2_new_sum

        self._convert_to_block_spins()

        q_molecule_new = copy.deepcopy(q_molecule)

        q_molecule_new.mo_onee_ints = self._h1[:int(self._h1.shape[0]/2), :int(self._h1.shape[1]/2)]

        # TODO calculation of mo_eri is still wrong, because of the doubling of the space.
        q_molecule_new.mo_eri_ints = self._h2[:int(self._h2.shape[0]/2), :int(self._h2.shape[1]/2),
                                              :int(self._h2.shape[2]/2), :int(self._h2.shape[3]/2)]

        q_molecule_new.energy_shift['ParticleHoleTransformer'] = identities_new_sum

        return q_molecule_new

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
        self._integral_transform(matrix)

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
        self._integral_transform(matrix)

    def _integral_transform(self, unitary_matrix):
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

    def _sort(self, seq):
        """
        Tool function for normal order, should not be used separately

        Args:
            seq (list): array

        Returns:
            list: integer e.g. swapped array, number of swaps
        """
        swap_counter = 0
        changed = True
        while changed:
            changed = False
            for i in range(len(seq) - 1):
                if seq[i] > seq[i + 1]:
                    swap_counter += 1
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
                    changed = True

        return seq, swap_counter

    def last_two_indices_swap(self, array_ind_two_body_term):
        """
        Swap 2 last indices of an array

        Args:
            array_ind_two_body_term (list): TBD

        Returns:
            list: TBD
        """
        swapped_indices = [0, 0, 0, 0]
        swapped_indices[0] = array_ind_two_body_term[0]
        swapped_indices[1] = array_ind_two_body_term[1]
        swapped_indices[2] = array_ind_two_body_term[3]
        swapped_indices[3] = array_ind_two_body_term[2]

        return swapped_indices

    def _normal_order_integrals(self, n_qubits, n_occupied, array_to_normal_order,
                                array_mapping, h1_old, h2_old,
                                h1_new, h2_new):
        """
        Given an operator and the rFs and rsgtu from Gaussian it produces new
        h1,h2,id_terms usable for the generation of the Hamiltonian in Pauli strings form.

        Args:
            n_qubits (int): number of qubits
            n_occupied (int): number of electrons (occupied orbitals)
            array_to_normal_order (list):  e.g. [i,j,k,l] indices of the term to normal order
            array_mapping (list): e.g. two body terms list  ['adag', 'adag', 'a', 'a'],
            single body terms list (list): ['adag', 'a']
            h1_old (numpy.ndarray):
                           e.g. rFs.dat (dim(rsgtu) = [n_qubits,n_qubits,n_qubits,n_qubits])
                           loaded with QuTip function (qutip.fileio.qload) or numpy.array
            h2_old (numpy.ndarray): e.g. rsgtu.dat (dim(rsgtu) = [n_qubits,n_qubits])
            h1_new (numpy.ndarray): e.g. numpy.zeros([n_qubits, n_qubits])
            h2_new (numpy.ndarray): e.g. numpy.zeros([n_qubits, n_qubits, n_qubits, n_qubits])

        Returns:
            Tuple(numpy.ndarray, numpy.ndarray, float): h1_new, h2_new, id_term
        """
        # TODO: extract these because they remain constant for all invocations of this method
        a_enum = np.arange(1, n_qubits+1)
        a_enum[n_occupied] *= -1
        adag_enum = a_enum * -1

        # We want to sort the occurrence of the `+` and `-` operators by their index (as given in
        # array_to_normal_order). Thus, we need to associate each value in this array with a unique
        # label which identifies both, the kind of operator, and it's position -> a_enum, adag_enum.
        array_to_sort = np.asarray([
            a_enum[val] if array_mapping[ind] == 'a' else adag_enum[val]
            for ind, val in enumerate(array_to_normal_order)
        ])

        array_sorted, swap_count = self._sort(array_to_sort)
        sign_no_term = (-1.) ** swap_count

        ind_ini_term = array_to_normal_order.copy()  # initial index array
        ind_no_term = np.asarray([abs(i) - 1 for i in array_sorted])  # normal-ordered index array
        mapping_no_term = ['a' if i in a_enum else 'adag' for i in array_sorted]

        i_i = 0
        j_j = 1
        k_k = 2
        l_l = 3

        id_term = 0.

        if len(array_to_normal_order) == 2:
            if ind_no_term[0] == ind_no_term[1]:
                if mapping_no_term == ['adag', 'a']:
                    temp_sign_h1 = float(1 * sign_no_term)

                    ind_old_h1 = [ind_ini_term[i_i], ind_ini_term[j_j]]
                    ind_new_h1 = [ind_no_term[i_i], ind_no_term[j_j]]

                    h1_new[ind_new_h1[0]][ind_new_h1[1]] \
                        += float(temp_sign_h1 * h1_old[ind_old_h1[0]][ind_old_h1[1]])

                elif mapping_no_term == ['a', 'adag']:
                    temp_sign_h1 = float(-1 * sign_no_term)

                    ind_old_h1 = [ind_ini_term[i_i], ind_ini_term[j_j]]
                    ind_new_h1 = [ind_no_term[j_j], ind_no_term[i_i]]

                    h1_new[ind_new_h1[0]][ind_new_h1[1]] \
                        += float(temp_sign_h1 * h1_old[ind_old_h1[0]][ind_old_h1[1]])

                    id_term += float(sign_no_term * h1_old[ind_old_h1[0]][ind_old_h1[1]])

            else:
                if mapping_no_term == ['adag', 'a']:
                    temp_sign_h1 = float(1 * sign_no_term)

                    ind_old_h1 = [ind_ini_term[i_i], ind_ini_term[j_j]]
                    ind_new_h1 = [ind_no_term[i_i], ind_no_term[j_j]]

                    h1_new[ind_new_h1[0]][ind_new_h1[1]] \
                        += float(temp_sign_h1 * h1_old[ind_old_h1[0]][ind_old_h1[1]])

                elif mapping_no_term == ['a', 'adag']:
                    temp_sign_h1 = float(-1 * sign_no_term)

                    ind_old_h1 = [ind_ini_term[i_i], ind_ini_term[j_j]]
                    ind_new_h1 = [ind_no_term[j_j], ind_no_term[i_i]]

                    h1_new[ind_new_h1[0]][ind_new_h1[1]] \
                        += float(temp_sign_h1 * h1_old[ind_old_h1[0]][ind_old_h1[1]])

        elif len(array_to_normal_order) == 4:
            if len(set(ind_no_term)) == 4:
                if mapping_no_term == ['adag', 'adag', 'a', 'a']:
                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[i_i], ind_no_term[j_j],
                                  ind_no_term[k_k], ind_no_term[l_l]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:
                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[i_i], ind_no_term[k_k],
                                  ind_no_term[j_j], ind_no_term[l_l]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:
                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[i_i], ind_no_term[l_l],
                                  ind_no_term[j_j], ind_no_term[k_k]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:
                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[j_j], ind_no_term[k_k],
                                  ind_no_term[i_i], ind_no_term[l_l]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:
                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[j_j], ind_no_term[l_l],
                                  ind_no_term[i_i], ind_no_term[k_k]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:
                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[k_k], ind_no_term[l_l],
                                  ind_no_term[i_i], ind_no_term[j_j]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR 1')

            elif len(set(ind_no_term)) == 3:

                if ind_no_term[0] == ind_no_term[1]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[l_l],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[k_k],
                                      ind_no_term[l_l]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[l_l],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[l_l],
                                      ind_no_term[k_k]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[k_k],
                                      ind_no_term[l_l],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    else:
                        print('ERROR 2')

                elif ind_no_term[0] == ind_no_term[2]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[l_l],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[j_j],
                                      ind_no_term[l_l]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[l_l],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[l_l],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[l_l],
                                      ind_no_term[j_j]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    else:
                        print('ERROR 3')

                elif ind_no_term[0] == ind_no_term[3]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[j_j],
                                      ind_no_term[k_k]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[k_k],
                                      ind_no_term[j_j]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]
                    else:
                        print('ERROR 4')

                elif ind_no_term[1] == ind_no_term[2]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[i_i],
                                      ind_no_term[l_l]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[l_l],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[l_l],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[l_l],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[l_l],
                                      ind_no_term[i_i]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    else:
                        print('ERROR 5')

                elif ind_no_term[1] == ind_no_term[3]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[k_k],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[i_i],
                                      ind_no_term[k_k]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[k_k],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[k_k],
                                      ind_no_term[i_i]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    else:
                        print('ERROR 6')

                elif ind_no_term[2] == ind_no_term[3]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[k_k],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[j_j],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[j_j],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[i_i],
                                      ind_no_term[j_j]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[j_j],
                                      ind_no_term[i_i]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[k_k],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    else:
                        print('ERROR 7')

                else:
                    print('ERROR 8')

            elif len(set(ind_no_term)) == 2:

                if ind_no_term[0] == ind_no_term[1] and \
                        ind_no_term[2] == ind_no_term[3]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[i_i],
                                      ind_no_term[i_i]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[k_k],
                                      ind_no_term[k_k]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1_1 = -1 * sign_no_term
                        temp_sign_h1_2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        coordinates_for_old_h1_term_1 = [ind_no_term[i_i],
                                                         ind_no_term[i_i]]
                        ind_old_h1_2 = [ind_no_term[k_k],
                                        ind_no_term[k_k]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[coordinates_for_old_h1_term_1[0]][coordinates_for_old_h1_term_1[1]] \
                            += 0.5 * temp_sign_h1_1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1_2[0]][ind_old_h1_2[1]] \
                            += 0.5 * temp_sign_h1_2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        id_term += 0.5 * sign_no_term * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[k_k],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]
                    else:
                        print('ERROR')

                elif ind_no_term[0] == ind_no_term[2] and \
                        ind_no_term[1] == ind_no_term[3]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[i_i],
                                      ind_no_term[i_i]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[j_j],
                                      ind_no_term[j_j]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1_1 = 1 * sign_no_term
                        temp_sign_h1_2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        coordinates_for_old_h1_term_1 = [ind_no_term[i_i],
                                                         ind_no_term[i_i]]
                        ind_old_h1_2 = [ind_no_term[j_j],
                                        ind_no_term[j_j]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[coordinates_for_old_h1_term_1[0]][coordinates_for_old_h1_term_1[1]] \
                            += 0.5 * temp_sign_h1_1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1_2[0]][ind_old_h1_2[1]] \
                            += 0.5 * temp_sign_h1_2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        id_term += - 0.5 * sign_no_term * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    else:
                        print('ERROR')

                elif ind_no_term[0] == ind_no_term[3] and \
                        ind_no_term[1] == ind_no_term[2]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[i_i],
                                      ind_no_term[i_i]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[j_j],
                                      ind_no_term[j_j]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1_1 = -1 * sign_no_term
                        temp_sign_h1_2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        coordinates_for_old_h1_term_1 = [ind_no_term[i_i],
                                                         ind_no_term[i_i]]
                        ind_old_h1_2 = [ind_no_term[j_j],
                                        ind_no_term[j_j]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[coordinates_for_old_h1_term_1[0]][coordinates_for_old_h1_term_1[1]] \
                            += 0.5 * temp_sign_h1_1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1_2[0]][ind_old_h1_2[1]] \
                            += 0.5 * temp_sign_h1_2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        id_term += 0.5 * sign_no_term * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]
                    else:
                        print('ERROR')

                elif ind_no_term[0] == ind_no_term[1] and \
                        ind_no_term[0] == ind_no_term[2]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:
                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1_1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        coordinates_for_old_h1_term_1 = [ind_no_term[i_i],
                                                         ind_no_term[l_l]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[coordinates_for_old_h1_term_1[0]][coordinates_for_old_h1_term_1[1]] \
                            += 0.5 * temp_sign_h1_1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[l_l],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[l_l]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1_1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[l_l],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        coordinates_for_old_h1_term_1 = [ind_no_term[l_l],
                                                         ind_no_term[i_i]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[coordinates_for_old_h1_term_1[0]][coordinates_for_old_h1_term_1[1]] \
                            += 0.5 * temp_sign_h1_1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[l_l],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    else:
                        print('ERROR')

                elif ind_no_term[0] == ind_no_term[1] and \
                        ind_no_term[0] == ind_no_term[3]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[i_i],
                                      ind_no_term[k_k]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[k_k]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[k_k],
                                      ind_no_term[i_i]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[k_k],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    else:
                        print('ERROR')

                elif ind_no_term[0] == ind_no_term[2] and \
                        ind_no_term[0] == ind_no_term[3]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[i_i],
                                      ind_no_term[j_j]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[j_j],
                                      ind_no_term[i_i]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]
                    else:
                        print('ERROR')

                elif ind_no_term[1] == ind_no_term[2] and \
                        ind_no_term[1] == ind_no_term[3]:

                    if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[i_i],
                                      ind_no_term[j_j]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[i_i],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                        temp_sign_h2 = -1 * sign_no_term
                        temp_sign_h1 = -1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        ind_old_h1 = [ind_no_term[j_j],
                                      ind_no_term[i_i]]

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                        h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                            += 0.5 * temp_sign_h1 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                        temp_sign_h2 = 1 * sign_no_term

                        ind_new_h2 = [ind_no_term[j_j],
                                      ind_no_term[j_j],
                                      ind_no_term[i_i],
                                      ind_no_term[j_j]]
                        ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                      ind_ini_term[2], ind_ini_term[3]]
                        ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                        h2_new[ind_new_h2[0]][ind_new_h2[1]][
                            ind_new_h2[2]][ind_new_h2[3]] \
                            += 0.5 * temp_sign_h2 * \
                            h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                    else:
                        print('ERROR')

                else:
                    print('ERROR')

            if len(set(ind_no_term)) == 1:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[i_i], ind_no_term[i_i],
                                  ind_no_term[i_i], ind_no_term[i_i]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[i_i], ind_no_term[i_i],
                                  ind_no_term[i_i], ind_no_term[i_i]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = self.last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR')

        return h1_new, h2_new, id_term
