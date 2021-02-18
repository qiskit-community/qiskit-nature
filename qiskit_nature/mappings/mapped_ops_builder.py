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
import itertools
import logging
import sys

import numpy as np
from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit_nature import QiskitNatureError
from qiskit_nature.bksf import bksf_mapping_ints
from qiskit_nature.fermionic_operator import logger


def mapping(map_type, num_modes, h1, h2, ph_trans_shift,
            threshold=0.00000001) -> WeightedPauliOperator:
    """
    Map fermionic operator to qubit operator.

    Using multiprocess to speedup the mapping, the improvement can be
    observed when h2 is a non-sparse matrix.

    Args:
        map_type (str): case-insensitive mapping type.
                        "jordan_wigner", "parity", "bravyi_kitaev", "bksf"
        threshold (float): threshold for Pauli simplification

    Returns:
        WeightedPauliOperator: create an Operator object in Paulis form.

    Raises:
        QiskitNatureError: if the `map_type` can not be recognized.
    """

    # ###################################################################
    # ###########   DEFINING MAPPED FERMIONIC OPERATORS    ##############
    # ###################################################################

    a_list = _define_mapped_ferm_ops(map_type, num_modes, h1, h2)
    if map_type == 'bksf':
        return a_list

    # ###################################################################
    # ###########    BUILDING THE MAPPED HAMILTONIAN     ################
    # ###################################################################

    pauli_list = WeightedPauliOperator(paulis=[])

    results = _map_one_body_terms(a_list, h1, num_modes, threshold)
    pauli_list = _extend_pauli_list(pauli_list, results, threshold)

    results = _map_two_body_terms(a_list, h2, num_modes, threshold)
    pauli_list = _extend_pauli_list(pauli_list, results, threshold)

    pauli_list = _apply_ph_trans_shift(num_modes, pauli_list, ph_trans_shift)

    return pauli_list


def _apply_ph_trans_shift(num_modes, pauli_list, ph_trans_shift):
    if ph_trans_shift is not None:
        pauli_term = [ph_trans_shift, Pauli.from_label('I' * num_modes)]
        pauli_list += WeightedPauliOperator(paulis=[pauli_term])
    return pauli_list


def _extend_pauli_list(pauli_list, results, threshold):
    for result in results:
        pauli_list += result
    pauli_list.chop(threshold=threshold)
    return pauli_list


def _map_one_body_terms(a_list, h1, num_modes, threshold):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Mapping one-body terms to Qubit Hamiltonian:")
        TextProgressBar(output_handler=sys.stderr)

    results = parallel_map(_one_body_mapping, [(h1[i, j], a_list[i], a_list[j])
                                               for i, j in
                                               itertools.product(range(num_modes), repeat=2)
                                               if h1[i, j] != 0],
                           task_args=(threshold,), num_processes=aqua_globals.num_processes)
    return results


def _map_two_body_terms(a_list, h2, num_modes, threshold):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Mapping two-body terms to Qubit Hamiltonian:")
        TextProgressBar(output_handler=sys.stderr)
    results = parallel_map(_two_body_mapping,
                           [(h2[i, j, k, m], a_list[i], a_list[j], a_list[k], a_list[m])
                            for i, j, k, m in itertools.product(range(num_modes), repeat=4)
                            if h2[i, j, k, m] != 0],
                           task_args=(threshold,), num_processes=aqua_globals.num_processes)
    return results


def _define_mapped_ferm_ops(map_type, num_modes, h1, h2) -> np.ndarray:
    map_type = map_type.lower()
    if map_type == 'jordan_wigner':
        a_list = _jordan_wigner_mode(num_modes)
    elif map_type == 'parity':
        a_list = _parity_mode(num_modes)
    elif map_type == 'bravyi_kitaev':
        a_list = _bravyi_kitaev_mode(num_modes)
    elif map_type == 'bksf':
        return bksf_mapping_ints(num_modes, h1, h2)
    else:
        raise QiskitNatureError('Please specify the supported modes: '
                                'jordan_wigner, parity, bravyi_kitaev, bksf')
    return a_list


def _two_body_mapping(h2_ijkm_a_ijkm, threshold):
    """
    Subroutine for two body mapping. We use the chemists notation
    for the two-body term, h2(i,j,k,m) adag_i adag_k a_m a_j.

    Args:
        h2_ijkm_a_ijkm (tuple): value of h2 at index (i,j,k,m),
                                pauli at index i, pauli at index j,
                                pauli at index k, pauli at index m
        threshold (float): threshold to remove a pauli

    Returns:
        WeightedPauliOperator: Operator for those paulis
    """
    h2_ijkm, a_i, a_j, a_k, a_m = h2_ijkm_a_ijkm
    pauli_list = []
    for alpha in range(2):
        for beta in range(2):
            for gamma in range(2):
                for delta in range(2):
                    pauli_prod_1 = Pauli.sgn_prod(a_i[alpha], a_k[beta])
                    pauli_prod_2 = Pauli.sgn_prod(pauli_prod_1[0], a_m[gamma])
                    pauli_prod_3 = Pauli.sgn_prod(pauli_prod_2[0], a_j[delta])

                    phase1 = pauli_prod_1[1] * pauli_prod_2[1] * pauli_prod_3[1]
                    phase2 = np.power(-1j, alpha + beta) * np.power(1j, gamma + delta)
                    pauli_term = [h2_ijkm / 16 * phase1 * phase2, pauli_prod_3[0]]
                    if np.absolute(pauli_term[0]) > threshold:
                        pauli_list.append(pauli_term)
    return WeightedPauliOperator(paulis=pauli_list)


def _one_body_mapping(h1_ij_aij, threshold):
    """
    Subroutine for one body mapping.

    Args:
        h1_ij_aij (tuple): value of h1 at index (i,j), pauli at index i, pauli at index j
        threshold (float): threshold to remove a pauli

    Returns:
        WeightedPauliOperator: Operator for those paulis
    """
    h1_ij, a_i, a_j = h1_ij_aij
    pauli_list = []
    for alpha in range(2):
        for beta in range(2):
            pauli_prod = Pauli.sgn_prod(a_i[alpha], a_j[beta])
            coeff = h1_ij / 4 * pauli_prod[1] * np.power(-1j, alpha) * np.power(1j, beta)
            pauli_term = [coeff, pauli_prod[0]]
            if np.absolute(pauli_term[0]) > threshold:
                pauli_list.append(pauli_term)
    return WeightedPauliOperator(paulis=pauli_list)


def _jordan_wigner_mode(n):
    r"""
    Jordan_Wigner mode.

    Each Fermionic Operator is mapped to 2 Pauli Operators, added together with the
    appropriate phase, i.e.:

    a_i\^\\dagger = Z\^i (X + iY) I\^(n-i-1) = (Z\^i X I\^(n-i-1)) + i (Z\^i Y I\^(n-i-1))
    a_i = Z\^i (X - iY) I\^(n-i-1)

    This is implemented by creating an array of tuples, each including two operators.
    The phase between two elements in a tuple is implicitly assumed, and added calculated at the
    appropriate time (see for example _one_body_mapping).

    Args:
        n (int): number of modes

    Returns:
        list[Tuple]: Pauli
    """
    a_list = []
    for i in range(n):
        a_z = np.asarray([1] * i + [0] + [0] * (n - i - 1), dtype=bool)
        a_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=bool)
        b_z = np.asarray([1] * i + [1] + [0] * (n - i - 1), dtype=bool)
        b_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=bool)
        a_list.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
    return a_list


def _parity_mode(n):
    """
    Parity mode.

    Args:
        n (int): number of modes

    Returns:
        list[Tuple]: Pauli
    """
    a_list = []
    for i in range(n):
        a_z = [0] * (i - 1) + [1] if i > 0 else []
        a_x = [0] * (i - 1) + [0] if i > 0 else []
        b_z = [0] * (i - 1) + [0] if i > 0 else []
        b_x = [0] * (i - 1) + [0] if i > 0 else []
        a_z = np.asarray(a_z + [0] + [0] * (n - i - 1), dtype=bool)
        a_x = np.asarray(a_x + [1] + [1] * (n - i - 1), dtype=bool)
        b_z = np.asarray(b_z + [1] + [0] * (n - i - 1), dtype=bool)
        b_x = np.asarray(b_x + [1] + [1] * (n - i - 1), dtype=bool)
        a_list.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
    return a_list


def _bravyi_kitaev_mode(n):
    """
    Bravyi-Kitaev mode.

    Args:
        n (int): number of modes

     Returns:
         numpy.ndarray: Array of mode indexes
    """

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

    def update_set(j, n: int):
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
        elif j >= n / 2 and j < n - 1:  # pylint: disable=chained-comparison
            indexes = np.append(indexes, flip_set(j - n / 2, n / 2) + n / 2)
        else:
            indexes = np.append(np.append(indexes, flip_set(
                j - n / 2, n / 2) + n / 2), n / 2 - 1)
        return indexes

    a_list = []
    # FIND BINARY SUPERSET SIZE
    bin_sup = 1
    # pylint: disable=comparison-with-callable
    while n > np.power(2, bin_sup):
        bin_sup += 1
    # DEFINE INDEX SETS FOR EVERY FERMIONIC MODE
    update_sets = []
    update_pauli = []

    parity_sets = []
    parity_pauli = []

    flip_sets = []

    remainder_sets = []
    remainder_pauli = []
    for j in range(n):

        update_sets.append(update_set(j, np.power(2, bin_sup)))
        update_sets[j] = update_sets[j][update_sets[j] < n]

        parity_sets.append(parity_set(j, np.power(2, bin_sup)))
        parity_sets[j] = parity_sets[j][parity_sets[j] < n]

        flip_sets.append(flip_set(j, np.power(2, bin_sup)))
        flip_sets[j] = flip_sets[j][flip_sets[j] < n]

        remainder_sets.append(np.setdiff1d(parity_sets[j], flip_sets[j]))

        update_pauli.append(Pauli(np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)))
        parity_pauli.append(Pauli(np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)))
        remainder_pauli.append(Pauli(np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)))
        for k in range(n):
            if np.in1d(k, update_sets[j]):
                update_pauli[j].update_x(True, k)
            if np.in1d(k, parity_sets[j]):
                parity_pauli[j].update_z(True, k)
            if np.in1d(k, remainder_sets[j]):
                remainder_pauli[j].update_z(True, k)

        x_j = Pauli(np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))
        x_j.update_x(True, j)
        y_j = Pauli(np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))
        y_j.update_z(True, j)
        y_j.update_x(True, j)
        a_list.append((update_pauli[j] * x_j * parity_pauli[j],
                       update_pauli[j] * y_j * remainder_pauli[j]))
    return a_list
