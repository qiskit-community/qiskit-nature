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

        h1_new_sum = np.zeros((n_qubits, n_qubits))
        h2_new_sum = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

        h2_old_matrix = -2 * h2_old_matrix.copy()
        h2_old_matrix = np.einsum('ijkl->iklj', h2_old_matrix.copy())

        # TODO: What is the point of doing this?!
        h1_old_matrix = h1_old_matrix.copy()

        # put labels of occupied orbitals in the list in interleaved spin convention
        n_occupied = [2 * a for a in range(num_alpha)] + [2 * b + 1 for b in range(num_beta)]
        # map the orbitals to creation and annihilation operators
        a_enum = np.arange(1, n_qubits+1)
        a_enum[n_occupied] *= -1
        adag_enum = a_enum * -1

        # TODO(bpark): add an explanatory comment for the below
        for r in range(n_qubits):
            for s in range(n_qubits):  # pylint: disable=invalid-name
                for i in n_occupied:
                    h1_old_matrix[r, s] += h2_old_matrix[r, i, s, i].copy() - \
                                           h2_old_matrix[r, i, i, s].copy()
        identities_new_sum = 0

        for i in range(n_qubits):
            for j in range(n_qubits):
                indices_1 = np.asarray((i, j))
                array_mapping_1 = '+-'

                h1_new_matrix = np.zeros((n_qubits, n_qubits))
                h2_new_matrix = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

                h1_new_matrix, h2_new_matrix, identities = self._normal_order_integrals(
                    a_enum, adag_enum, indices_1, array_mapping_1,
                    h1_old_matrix, h2_old_matrix, h1_new_matrix, h2_new_matrix)

                h1_new_sum += h1_new_matrix
                h2_new_sum += h2_new_matrix
                identities_new_sum += identities

        for i in range(n_qubits):
            for j in range(n_qubits):
                for k in range(n_qubits):
                    for l in range(n_qubits):  # pylint: disable=invalid-name
                        indices_2 = np.asarray((i, j, k, l))
                        array_mapping_2 = '++--'

                        h1_new_matrix = np.zeros((n_qubits, n_qubits))
                        h2_new_matrix = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

                        h1_new_matrix, h2_new_matrix, identities = self._normal_order_integrals(
                            a_enum, adag_enum, indices_2, array_mapping_2,
                            h1_old_matrix, h2_old_matrix, h1_new_matrix, h2_new_matrix)

                        h1_new_sum += h1_new_matrix
                        h2_new_sum += h2_new_matrix
                        identities_new_sum += identities

        h2_new_sum = np.einsum('ikmj->ijkm', h2_new_sum)

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
        matrix = np.zeros_like(self._h1)
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
        matrix = np.zeros_like(self._h1)
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

    def _bubble_sort_with_swap_count(self, seq):
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

    def _normal_order_integrals(self, a_enum, adag_enum, array_to_normal_order,
                                array_mapping, h1_old, h2_old,
                                h1_new, h2_new):
        """
        Given an operator and the rFs and rsgtu from Gaussian it produces new
        h1,h2,id_terms usable for the generation of the Hamiltonian in Pauli strings form.

        Args:
            a_enum (np.ndarray): TODO
            adag_enum (np.ndarray): TODO
            array_to_normal_order (list):  e.g. [i,j,k,l] indices of the term to normal order
            array_mapping (str): e.g. two body terms '++--', or single body terms '+-'
            h1_old (numpy.ndarray):
                           e.g. rFs.dat (dim(rsgtu) = [n_qubits,n_qubits,n_qubits,n_qubits])
                           loaded with QuTip function (qutip.fileio.qload) or numpy.array
            h2_old (numpy.ndarray): e.g. rsgtu.dat (dim(rsgtu) = [n_qubits,n_qubits])
            h1_new (numpy.ndarray): e.g. numpy.zeros([n_qubits, n_qubits])
            h2_new (numpy.ndarray): e.g. numpy.zeros([n_qubits, n_qubits, n_qubits, n_qubits])

        Returns:
            Tuple(numpy.ndarray, numpy.ndarray, float): h1_new, h2_new, id_term
        """
        # We want to sort the occurrence of the `+` and `-` operators by their index (as given in
        # array_to_normal_order). Thus, we need to associate each value in this array with a unique
        # label which identifies both, the kind of operator, and it's position -> a_enum, adag_enum.
        array_to_sort = np.asarray([
            a_enum[val] if array_mapping[ind] == '-' else adag_enum[val]
            for ind, val in enumerate(array_to_normal_order)
        ])

        array_sorted, swap_count = self._bubble_sort_with_swap_count(array_to_sort)
        sign_no_term = (-1.) ** swap_count

        ind_ini_term = array_to_normal_order.copy()  # initial index array
        ind_no_term = np.asarray([abs(i) - 1 for i in array_sorted])  # normal-ordered index array
        mapping_no_term = ''.join(['-' if i in a_enum else '+' for i in array_sorted])

        # This is a list in order to make it available in the inlined utility methods below.
        id_term = [0.]

        # This utility method is inlined in order to avoid a lot of data-shuffling
        def update_h1(indices, sign_flip=False, update_id_term=False):
            ind_old = (ind_ini_term[0], ind_ini_term[1])
            ind_new = tuple(ind_no_term[np.asarray(indices)])

            h1_new[ind_new] += (-1.) ** sign_flip * sign_no_term * h1_old[ind_old]

            if update_id_term:
                id_term[0] += float(sign_no_term * h1_old[ind_old])

        # This utility method is inlined in order to avoid a lot of data-shuffling
        def update_h2(indices, sign_flip=False, indices_h1=None, sign_flip_h1=False,
                      update_id_term=False, sign_flip_id=False):
            ind_old = tuple(ind_ini_term[np.asarray((0, 1, 3, 2))])
            ind_new = tuple(ind_no_term[np.asarray(indices)])

            h2_new[ind_new] += 0.5 * (-1.) ** sign_flip * sign_no_term * h2_old[ind_old]

            if indices_h1:
                if isinstance(indices_h1, tuple):
                    indices_h1 = [indices_h1]
                for ind_h1 in indices_h1:
                    ind_old_ = tuple(ind_no_term[np.asarray(ind_h1)])

                    h1_new[ind_old_] += 0.5 * (-1.) ** sign_flip_h1 * sign_no_term * h2_old[ind_old]

            if update_id_term:
                id_term[0] += 0.5 * (-1.) ** sign_flip_id * sign_no_term * h2_old[ind_old]

        if len(array_to_normal_order) == 2:
            if ind_no_term[0] == ind_no_term[1]:
                if mapping_no_term == '+-':
                    update_h1((0, 1))
                elif mapping_no_term == '-+':
                    update_h1((1, 0), sign_flip=True, update_id_term=True)
            else:
                if mapping_no_term == '+-':
                    update_h1((0, 1))
                elif mapping_no_term == '-+':
                    update_h1((1, 0), sign_flip=True)

        elif len(array_to_normal_order) == 4:
            if len(set(ind_no_term)) == 4:
                if mapping_no_term == '++--':
                    update_h2((0, 1, 2, 3))
                elif mapping_no_term == '+-+-':
                    update_h2((0, 2, 1, 3), sign_flip=True)
                elif mapping_no_term == '+--+':
                    update_h2((0, 3, 1, 2))
                elif mapping_no_term == '-++-':
                    update_h2((1, 2, 0, 3))
                elif mapping_no_term == '-+-+':
                    update_h2((1, 3, 0, 2), sign_flip=True)
                elif mapping_no_term == '--++':
                    update_h2((2, 3, 0, 1))
                else:
                    print('ERROR 1')

            elif len(set(ind_no_term)) == 3:
                if ind_no_term[0] == ind_no_term[1]:
                    if mapping_no_term == '++--':
                        update_h2((0, 0, 2, 3))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 2, 0, 3), sign_flip=True)
                    elif mapping_no_term == '+--+':
                        update_h2((0, 3, 0, 2))
                    elif mapping_no_term == '-++-':
                        update_h2((0, 2, 0, 3), indices_h1=(2, 3))
                    elif mapping_no_term == '-+-+':
                        update_h2((0, 3, 0, 2), sign_flip=True,
                                  indices_h1=(3, 2), sign_flip_h1=True)
                    elif mapping_no_term == '--++':
                        update_h2((2, 3, 0, 0))
                    else:
                        print('ERROR 2')

                elif ind_no_term[0] == ind_no_term[2]:
                    if mapping_no_term == '++--':
                        update_h2((0, 1, 0, 3))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 0, 1, 3), sign_flip=True)
                    elif mapping_no_term == '+--+':
                        update_h2((0, 3, 1, 0))
                    elif mapping_no_term == '-++-':
                        update_h2((1, 0, 0, 3),
                                  indices_h1=(1, 3), sign_flip_h1=True)
                    elif mapping_no_term == '-+-+':
                        update_h2((1, 3, 1, 3), sign_flip=True)
                    elif mapping_no_term == '--++':
                        update_h2((0, 3, 0, 1), indices_h1=(3, 1))
                    else:
                        print('ERROR 3')

                elif ind_no_term[0] == ind_no_term[3]:
                    if mapping_no_term == '++--':
                        update_h2((0, 1, 2, 0))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 2, 1, 0), sign_flip=True)
                    elif mapping_no_term == '+--+':
                        update_h2((0, 0, 1, 2))
                    elif mapping_no_term == '-++-':
                        update_h2((1, 2, 0, 0))
                    elif mapping_no_term == '-+-+':
                        update_h2((1, 0, 0, 2), sign_flip=True,
                                  indices_h1=(1, 2))
                    elif mapping_no_term == '--++':
                        update_h2((2, 0, 0, 1),
                                  indices_h1=(2, 1), sign_flip_h1=True)
                    else:
                        print('ERROR 4')

                elif ind_no_term[1] == ind_no_term[2]:
                    if mapping_no_term == '++--':
                        update_h2((0, 1, 1, 3))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 1, 1, 3), sign_flip=True,
                                  indices_h1=(0, 3))
                    elif mapping_no_term == '+--+':
                        update_h2((0, 3, 1, 1))
                    elif mapping_no_term == '-++-':
                        update_h2((1, 1, 0, 3))
                    elif mapping_no_term == '-+-+':
                        update_h2((1, 3, 0, 1), sign_flip=True)
                    elif mapping_no_term == '--++':
                        update_h2((1, 3, 0, 1),
                                  indices_h1=(3, 0), sign_flip_h1=True)
                    else:
                        print('ERROR 5')

                elif ind_no_term[1] == ind_no_term[3]:
                    if mapping_no_term == '++--':
                        update_h2((0, 1, 2, 1))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 2, 1, 1), sign_flip=True)
                    elif mapping_no_term == '+--+':
                        update_h2((0, 1, 1, 2),
                                  indices_h1=(0, 2), sign_flip_h1=True)
                    elif mapping_no_term == '-++-':
                        update_h2((1, 2, 0, 1))
                    elif mapping_no_term == '-+-+':
                        update_h2((1, 1, 0, 2), sign_flip=True)
                    elif mapping_no_term == '--++':
                        update_h2((2, 1, 0, 1), indices_h1=(2, 0))
                    else:
                        print('ERROR 6')

                elif ind_no_term[2] == ind_no_term[3]:
                    if mapping_no_term == '++--':
                        update_h2((0, 1, 2, 2))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 2, 1, 2), sign_flip=True)
                    elif mapping_no_term == '+--+':
                        update_h2((0, 2, 1, 2), indices_h1=(0, 1))
                    elif mapping_no_term == '-++-':
                        update_h2((1, 2, 0, 2))
                    elif mapping_no_term == '-+-+':
                        update_h2((1, 2, 0, 2), sign_flip=True,
                                  indices_h1=(1, 0), sign_flip_h1=True)
                    elif mapping_no_term == '--++':
                        update_h2((2, 2, 0, 1))
                    else:
                        print('ERROR 7')
                else:
                    print('ERROR 8')

            elif len(set(ind_no_term)) == 2:

                if ind_no_term[0] == ind_no_term[1] and ind_no_term[2] == ind_no_term[3]:

                    if mapping_no_term == '++--':
                        update_h2((0, 0, 2, 2))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 2, 0, 2), sign_flip=True)
                    elif mapping_no_term == '+--+':
                        update_h2((0, 2, 2, 0), sign_flip=True,
                                  indices_h1=(0, 0))
                    elif mapping_no_term == '-++-':
                        update_h2((0, 2, 0, 2), indices_h1=(2, 2))
                    elif mapping_no_term == '-+-+':
                        update_h2((0, 2, 0, 2), sign_flip=True,
                                  indices_h1=[(0, 0), (2, 2)], sign_flip_h1=True,
                                  update_id_term=True)
                    elif mapping_no_term == '--++':
                        update_h2((2, 2, 0, 0))
                    else:
                        print('ERROR')

                elif ind_no_term[0] == ind_no_term[2] and ind_no_term[1] == ind_no_term[3]:
                    if mapping_no_term == '++--':
                        update_h2((0, 1, 0, 1))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 0, 1, 1), sign_flip=True)
                    elif mapping_no_term == '+--+':
                        update_h2((0, 1, 1, 0),
                                  indices_h1=(0, 0), sign_flip_h1=True)
                    elif mapping_no_term == '-++-':
                        update_h2((1, 0, 0, 1),
                                  indices_h1=(1, 1), sign_flip_h1=True)
                    elif mapping_no_term == '-+-+':
                        update_h2((1, 1, 0, 0), sign_flip=True)
                    elif mapping_no_term == '--++':
                        update_h2((0, 1, 0, 1), indices_h1=[(0, 0), (1, 1)],
                                  update_id_term=True, sign_flip_id=True)
                    else:
                        print('ERROR')

                elif ind_no_term[0] == ind_no_term[3] and ind_no_term[1] == ind_no_term[2]:
                    if mapping_no_term == '++--':
                        update_h2((0, 1, 1, 0))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 1, 1, 0), sign_flip=True,
                                  indices_h1=(0, 0))
                    elif mapping_no_term == '+--+':
                        update_h2((0, 0, 1, 1))
                    elif mapping_no_term == '-++-':
                        update_h2((1, 1, 0, 0))
                    elif mapping_no_term == '-+-+':
                        update_h2((1, 0, 0, 1), sign_flip=True,
                                  indices_h1=(1, 1))
                    elif mapping_no_term == '--++':
                        update_h2((1, 0, 0, 1),
                                  indices_h1=[(0, 0), (1, 1)], sign_flip_h1=True,
                                  update_id_term=True)
                    else:
                        print('ERROR')

                elif ind_no_term[0] == ind_no_term[1] and ind_no_term[0] == ind_no_term[2]:
                    if mapping_no_term == '++--':
                        update_h2((0, 0, 0, 3))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 0, 0, 3), sign_flip=True,
                                  indices_h1=(0, 3))
                    elif mapping_no_term == '+--+':
                        update_h2((0, 3, 0, 0))
                    elif mapping_no_term == '-++-':
                        update_h2((0, 0, 0, 3))
                    elif mapping_no_term == '-+-+':
                        update_h2((0, 3, 0, 0), sign_flip=True,
                                  indices_h1=(3, 0), sign_flip_h1=True)
                    elif mapping_no_term == '--++':
                        update_h2((0, 3, 0, 0))
                    else:
                        print('ERROR')

                elif ind_no_term[0] == ind_no_term[1] and ind_no_term[0] == ind_no_term[3]:
                    if mapping_no_term == '++--':
                        update_h2((0, 0, 0, 2))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 0, 0, 2), sign_flip=True,
                                  indices_h1=(0, 2))
                    elif mapping_no_term == '+--+':
                        update_h2((0, 2, 0, 0))
                    elif mapping_no_term == '-++-':
                        update_h2((0, 0, 0, 2))
                    elif mapping_no_term == '-+-+':
                        update_h2((0, 2, 0, 0), sign_flip=True,
                                  indices_h1=(2, 0), sign_flip_h1=True)
                    elif mapping_no_term == '--++':
                        update_h2((0, 2, 0, 0))
                    else:
                        print('ERROR')

                elif ind_no_term[0] == ind_no_term[2] and ind_no_term[0] == ind_no_term[3]:
                    if mapping_no_term == '++--':
                        update_h2((0, 1, 0, 0))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 0, 1, 0), sign_flip=True)
                    elif mapping_no_term == '+--+':
                        update_h2((0, 0, 1, 0), indices_h1=(0, 1))
                    elif mapping_no_term == '-++-':
                        update_h2((1, 0, 0, 0), sign_flip=True,
                                  indices_h1=(1, 0), sign_flip_h1=True)
                    elif mapping_no_term == '-+-+':
                        update_h2((1, 0, 0, 0), sign_flip=True)
                    elif mapping_no_term == '--++':
                        update_h2((0, 0, 0, 1))
                    else:
                        print('ERROR')

                elif ind_no_term[1] == ind_no_term[2] and ind_no_term[1] == ind_no_term[3]:
                    if mapping_no_term == '++--':
                        update_h2((0, 1, 1, 1))
                    elif mapping_no_term == '+-+-':
                        update_h2((0, 1, 1, 1), sign_flip=True,
                                  indices_h1=(0, 1))
                    elif mapping_no_term == '+--+':
                        update_h2((0, 1, 1, 1))
                    elif mapping_no_term == '-++-':
                        update_h2((1, 1, 0, 1))
                    elif mapping_no_term == '-+-+':
                        update_h2((1, 1, 0, 1), sign_flip=True,
                                  indices_h1=(1, 0), sign_flip_h1=True)
                    elif mapping_no_term == '--++':
                        update_h2((1, 1, 0, 1))
                    else:
                        print('ERROR')
                else:
                    print('ERROR')

            if len(set(ind_no_term)) == 1:
                if mapping_no_term == '++--':
                    update_h2((0, 0, 0, 0))
                elif mapping_no_term == '--++':
                    update_h2((0, 0, 0, 0))
                else:
                    print('ERROR')

        return h1_new, h2_new, id_term[0]
