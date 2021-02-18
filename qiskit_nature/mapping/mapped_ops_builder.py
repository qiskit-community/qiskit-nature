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
from qiskit_nature.mapping.enums.ferm_mapping_type_enum import FermionicQubitMappingType
from qiskit_nature.mapping.mode_calculators.bravyi_kitaev_mode import bravyi_kitaev_mode
from qiskit_nature.mapping.mode_calculators.parity_mode import parity_mode
from qiskit_nature.mapping.mode_calculators.jordan_wigner_mode import jordan_wigner_mode


def mapping(map_type: str, num_modes: int, h_1: np.ndarray, h_2: np.ndarray, ph_trans_shift: float,
            threshold: float = 0.00000001) -> WeightedPauliOperator:
    """
    Map fermionic operator to qubit operator.

    Using multiprocess to speedup the mapping, the improvement can be
    observed when h2 is a non-sparse matrix.

    Args:
        map_type (str): case-insensitive mapping type.
                        "jordan_wigner", "parity", "bravyi_kitaev", "bksf"
        num_modes (int): number of modes
        h_1 (np.ndarray): one-body integrals
        h_2 (np.ndarray): two-body integrals
        ph_trans_shift (float): energy shift caused by particle hole transformation
        threshold (float): threshold for Pauli simplification

    Returns:
        WeightedPauliOperator: create an Operator object in Paulis form.

    Raises:
        QiskitNatureError: if the `map_type` can not be recognized.
    """
    if map_type == FermionicQubitMappingType.BKSF.value:
        return bksf_mapping_ints(num_modes, h_1, h_2)

    a_list = _define_mapped_ferm_ops(map_type, num_modes)

    pauli_list = WeightedPauliOperator(paulis=[])

    results = _map_one_body_terms(a_list, h_1, num_modes, threshold)
    pauli_list = _extend_pauli_list(pauli_list, results, threshold)

    results = _map_two_body_terms(a_list, h_2, num_modes, threshold)
    pauli_list = _extend_pauli_list(pauli_list, results, threshold)

    pauli_list = _apply_ph_trans_shift(num_modes, pauli_list, ph_trans_shift)

    return pauli_list


def _apply_ph_trans_shift(num_modes: int, pauli_list, ph_trans_shift: float):
    if ph_trans_shift is not None:
        pauli_term = [ph_trans_shift, Pauli.from_label('I' * num_modes)]
        pauli_list += WeightedPauliOperator(paulis=[pauli_term])
    return pauli_list


def _extend_pauli_list(pauli_list, results, threshold: float):
    for result in results:
        pauli_list += result
    pauli_list.chop(threshold=threshold)
    return pauli_list


def _map_one_body_terms(a_list, h_1: np.ndarray, num_modes: int, threshold: float):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Mapping one-body terms to Qubit Hamiltonian:")
        TextProgressBar(output_handler=sys.stderr)

    results = parallel_map(_one_body_mapping, [(h_1[i, j], a_list[i], a_list[j])
                                               for i, j in
                                               itertools.product(range(num_modes), repeat=2)
                                               if h_1[i, j] != 0],
                           task_args=(threshold,), num_processes=aqua_globals.num_processes)
    return results


def _map_two_body_terms(a_list, h_2: np.ndarray, num_modes: int, threshold: float):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Mapping two-body terms to Qubit Hamiltonian:")
        TextProgressBar(output_handler=sys.stderr)
    results = parallel_map(_two_body_mapping,
                           [(h_2[i, j, k, m], a_list[i], a_list[j], a_list[k], a_list[m])
                            for i, j, k, m in itertools.product(range(num_modes), repeat=4)
                            if h_2[i, j, k, m] != 0],
                           task_args=(threshold,), num_processes=aqua_globals.num_processes)
    return results


def _define_mapped_ferm_ops(map_type, num_modes: int) -> np.ndarray:
    map_type = map_type.lower()
    if map_type == FermionicQubitMappingType.JORDAN_WIGNER.value:
        a_list = jordan_wigner_mode(num_modes)
    elif map_type == FermionicQubitMappingType.PARITY.value:
        a_list = parity_mode(num_modes)
    elif map_type == FermionicQubitMappingType.BRAVYI_KITAEV.value:
        a_list = bravyi_kitaev_mode(num_modes)
    else:
        raise QiskitNatureError('Please specify the supported modes: '
                                'jordan_wigner, parity, bravyi_kitaev, bksf')
    return a_list


def _one_body_mapping(h1_ij_aij, threshold: float):
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


def _two_body_mapping(h2_ijkm_a_ijkm, threshold: float):
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
    for alpha, beta, gamma, delta in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):
        pauli_prod_1 = Pauli.sgn_prod(a_i[alpha], a_k[beta])
        pauli_prod_2 = Pauli.sgn_prod(pauli_prod_1[0], a_m[gamma])
        pauli_prod_3 = Pauli.sgn_prod(pauli_prod_2[0], a_j[delta])

        phase1 = pauli_prod_1[1] * pauli_prod_2[1] * pauli_prod_3[1]
        phase2 = np.power(-1j, alpha + beta) * np.power(1j, gamma + delta)
        pauli_term = [h2_ijkm / 16 * phase1 * phase2, pauli_prod_3[0]]
        if np.absolute(pauli_term[0]) > threshold:
            pauli_list.append(pauli_term)
    return WeightedPauliOperator(paulis=pauli_list)
