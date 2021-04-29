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

"""The Bravyi-Kitaev Super Fast Mapper."""

import copy
import itertools
import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli, SparsePauliOp

from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit.chemistry import FermionicOperator

from .fermionic_mapper import FermionicMapper


def _pauli_id(n_qubits):
    return SparsePauliOp.from_operator(Pauli((np.zeros(n_qubits, dtype=bool), np.zeros(n_qubits, dtype=bool))))

def _one_body(edge_list, p, q, h1_pq):  # pylint: disable=invalid-name
    """
    Map the term a^\\dagger_p a_q + a^\\dagger_q a_p to qubit operator.
    Args:
        edge_list (numpy.ndarray): 2xE matrix, each indicates (from, to) pair
        p (int): index of the one body term
        q (int): index of the one body term
        h1_pq (complex): coefficient of the one body term at (p, q)

    Return:
        Pauli: mapped qubit operator
    """
    # Handle off-diagonal terms.
    if p != q:
        a_i, b_i = sorted([p, q])
        b_a = edge_operator_bi(edge_list, a_i)
        b_b = edge_operator_bi(edge_list, b_i)
        a_ab = edge_operator_aij(edge_list, a_i, b_i)
        qubit_op = a_ab * b_b + b_a * a_ab
        final_coeff = -1j * 0.5

    # Handle diagonal terms.
    else:
        b_p = edge_operator_bi(edge_list, p)
        id_op = _pauli_id(edge_list.shape[1])
        qubit_op = id_op - b_p
        final_coeff = 0.5

    qubit_op = (final_coeff * h1_pq) * qubit_op
    qubit_op.simplify()
    return qubit_op

def _two_body(edge_list, p, q, r, s, h2_pqrs):  # pylint: disable=invalid-name
    """
    Map the term a^\\dagger_p a^\\dagger_q a_r a_s + h.c. to qubit operator.

    Args:
        edge_list (numpy.ndarray): 2xE matrix, each indicates (from, to) pair
        p (int): index of the two body term
        q (int): index of the two body term
        r (int): index of the two body term
        s (int): index of the two body term
        h2_pqrs (complex): coefficient of the two body term at (p, q, r, s)

    Returns:
        Pauli: mapped qubit operator
    """
    # Handle case of four unique indices.
    id_op = _pauli_id(edge_list.shape[1])
    final_coeff = 1.0

    if len(set([p, q, r, s])) == 4:
        b_p = edge_operator_bi(edge_list, p)
        b_q = edge_operator_bi(edge_list, q)
        b_r = edge_operator_bi(edge_list, r)
        b_s = edge_operator_bi(edge_list, s)
        a_pq = edge_operator_aij(edge_list, p, q)
        a_rs = edge_operator_aij(edge_list, r, s)
        a_pq = -a_pq if q < p else a_pq
        a_rs = -a_rs if s < r else a_rs

        qubit_op = (a_pq * a_rs) * (-id_op - b_p * b_q + b_p * b_r
                                    + b_p * b_s + b_q * b_r + b_q * b_s
                                    - b_r * b_s - b_p * b_q * b_r * b_s)
        final_coeff = 0.125

    # Handle case of three unique indices.
    elif len(set([p, q, r, s])) == 3:
        b_p = edge_operator_bi(edge_list, p)
        b_q = edge_operator_bi(edge_list, q)
        if p == r:
            b_s = edge_operator_bi(edge_list, s)
            a_qs = edge_operator_aij(edge_list, q, s)
            a_qs = -a_qs if s < q else a_qs
            qubit_op = (a_qs * b_s + b_q * a_qs) * (id_op - b_p)
            final_coeff = 1j * 0.25
        elif p == s:
            b_r = edge_operator_bi(edge_list, r)
            a_qr = edge_operator_aij(edge_list, q, r)
            a_qr = -a_qr if r < q else a_qr
            qubit_op = (a_qr * b_r + b_q * a_qr) * (id_op - b_p)
            final_coeff = 1j * -0.25
        elif q == r:
            b_s = edge_operator_bi(edge_list, s)
            a_ps = edge_operator_aij(edge_list, p, s)
            a_ps = -a_ps if s < p else a_ps
            qubit_op = (a_ps * b_s + b_p * a_ps) * (id_op - b_q)
            final_coeff = 1j * -0.25
        elif q == s:
            b_r = edge_operator_bi(edge_list, r)
            a_pr = edge_operator_aij(edge_list, p, r)
            a_pr = -a_pr if r < p else a_pr
            qubit_op = (a_pr * b_r + b_p * a_pr) * (id_op - b_q)
            final_coeff = 1j * 0.25
        else:
            pass

    # Handle case of two unique indices.
    elif len(set([p, q, r, s])) == 2:
        b_p = edge_operator_bi(edge_list, p)
        b_q = edge_operator_bi(edge_list, q)
        qubit_op = (id_op - b_p) * (id_op - b_q)
        if p == s:
            final_coeff = 0.25
        else:
            final_coeff = -0.25
    else:
        pass

    qubit_op = (final_coeff * h2_pqrs) * qubit_op
    qubit_op.simplify()
    return qubit_op

def bravyi_kitaev_fast_edge_list(fer_op):
    """
    Construct edge list required for the bksf algorithm.

    Args:
        fer_op (FeriomicOperator): the fermionic operator in the second quantized form

    Returns:
        numpy.ndarray: edge_list, a 2xE matrix, where E is total number of edge
                        and each pair denotes (from, to)
    """
    h_1 = fer_op.h1
    h_2 = fer_op.h2
    modes = fer_op.modes
    edge_matrix = np.zeros((modes, modes), dtype=bool)

    for p, q in itertools.product(range(modes), repeat=2):  # pylint: disable=invalid-name

        if h_1[p, q] != 0.0 and p >= q:
            edge_matrix[p, q] = True

        for r, s in itertools.product(range(modes), repeat=2):  # pylint: disable=invalid-name
            if h_2[p, q, r, s] == 0.0:  # skip zero terms
                continue

            # Identify and skip one of the complex conjugates.
            if [p, q, r, s] != [s, r, q, p]:
                if len(set([p, q, r, s])) == 4:
                    if min(r, s) < min(p, q):
                        continue
                elif p != r and q < p:
                    continue

            # Handle case of four unique indices.
            if len(set([p, q, r, s])) == 4:
                if p >= q:
                    edge_matrix[p, q] = True
                    a_i, b = sorted([r, s])
                    edge_matrix[b, a_i] = True

            # Handle case of three unique indices.
            elif len(set([p, q, r, s])) == 3:
                # Identify equal tensor factors.
                if p == r:
                    a_i, b = sorted([q, s])
                elif p == s:
                    a_i, b = sorted([q, r])
                elif q == r:
                    a_i, b = sorted([p, s])
                elif q == s:
                    a_i, b = sorted([p, r])
                else:
                    continue
                edge_matrix[b, a_i] = True

    edge_list = np.asarray(np.nonzero(np.triu(edge_matrix.T) ^ np.diag(np.diag(edge_matrix.T))))
    return edge_list


def edge_operator_aij(edge_list, i, j):
    """Calculate the edge operator A_ij.

    The definitions used here are consistent with arXiv:quant-ph/0003137

    Args:
        edge_list (numpy.ndarray): a 2xE matrix, where E is total number of edge
                                    and each pair denotes (from, to)
        i (int): specifying the edge operator A
        j (int): specifying the edge operator A

    Returns:
        Pauli: qubit operator
    """
    v = np.zeros(edge_list.shape[1])
    w = np.zeros(edge_list.shape[1])

    position_ij = -1
    qubit_position_i = np.asarray(np.where(edge_list == i))

    for edge_index in range(edge_list.shape[1]):
        if set((i, j)) == set(edge_list[:, edge_index]):
            position_ij = edge_index
            break

    w[position_ij] = 1

    for edge_index in range(qubit_position_i.shape[1]):
        i_i, j_j = qubit_position_i[:, edge_index]
        i_i = 1 if i_i == 0 else 0  # int(not(i_i))
        if edge_list[i_i][j_j] < j:
            v[j_j] = 1

    qubit_position_j = np.asarray(np.where(edge_list == j))
    for edge_index in range(qubit_position_j.shape[1]):
        i_i, j_j = qubit_position_j[:, edge_index]
        i_i = 1 if i_i == 0 else 0  # int(not(i_i))
        if edge_list[i_i][j_j] < i:
            v[j_j] = 1

    qubit_op = Pauli((v, w))
    return SparsePauliOp.from_operator(qubit_op)

def edge_operator_bi(edge_list, i):
    """Calculate the edge operator B_i.

    The definitions used here are consistent with arXiv:quant-ph/0003137

    Args:
        edge_list (numpy.ndarray): a 2xE matrix, where E is total number of edge
                                    and each pair denotes (from, to)
        i (int): index for specifying the edge operator B.

    Returns:
        Pauli: qubit operator
    """
    qubit_position_matrix = np.asarray(np.where(edge_list == i))
    qubit_position = qubit_position_matrix[1]
    v = np.zeros(edge_list.shape[1])
    w = np.zeros(edge_list.shape[1])
    v[qubit_position] = 1
    qubit_op = Pauli((v, w))
    return SparsePauliOp.from_operator(qubit_op)


class BravyiKitaevSFMapper(FermionicMapper):
    """The Bravyi-Kitaev super fast fermion-to-qubit mapping. """

    def map(self, second_q_op: FermionicOperator) -> PauliSumOp:
        second_q_op = copy.deepcopy(second_q_op)
        # bksf mapping works with the 'physicist' notation.
        second_q_op.h2 = np.einsum('ijkm->ikmj', second_q_op.h2)
        modes = second_q_op.modes
        # Initialize qubit operator as constant.
        qubit_op = None # SparsePauliOp.from_operator(Pauli(([False], [False])))
        edge_list = bravyi_kitaev_fast_edge_list(second_q_op)
        # Loop through all indices.
        for p in range(modes):  # pylint: disable=invalid-name
            for q in range(modes):
                # Handle one-body terms.
                h1_pq = second_q_op.h1[p, q]

                if h1_pq != 0.0 and p >= q:
                    if qubit_op is None:
                        qubit_op = _one_body(edge_list, p, q, h1_pq)
                    else:
                        qubit_op += _one_body(edge_list, p, q, h1_pq)

                # Keep looping for the two-body terms.
                for r in range(modes):
                    for s in range(modes):  # pylint: disable=invalid-name
                        h2_pqrs = second_q_op.h2[p, q, r, s]

                        # Skip zero terms.
                        if (h2_pqrs == 0.0) or (p == q) or (r == s):
                            continue

                        # Identify and skip one of the complex conjugates.
                        if [p, q, r, s] != [s, r, q, p]:
                            if len(set([p, q, r, s])) == 4:
                                if min(r, s) < min(p, q):
                                    continue
                            # Handle case of 3 unique indices
                            elif len(set([p, q, r, s])) == 3:
                                qubit_op += _two_body(edge_list, p, q, r, s, 0.5 * h2_pqrs)
                                continue
                            elif p != r and q < p:
                                continue

                        qubit_op += _two_body(edge_list, p, q, r, s, h2_pqrs)

        qubit_op.simplify()
        return PauliSumOp(qubit_op)
