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

""" Fermionic operator builder. """

import itertools

import numpy as np

from qiskit_nature.drivers.qmolecule import QMolecule
from qiskit_nature.operators import FermionicOp


def build_fermionic_op(q_molecule: QMolecule) -> FermionicOp:
    """
    Builds a fermionic operator based on a QMolecule object.

    Args:
        q_molecule (QMolecule): QMolecule instance with 1- and/or 2-body integrals.

    Returns:
        FermionicOp: FermionicOp built from a QMolecule object.
    """

    one_body_ints = q_molecule.one_body_integrals
    two_body_ints = q_molecule.two_body_integrals

    return build_ferm_op_from_ints(one_body_ints, two_body_ints)


def build_ferm_op_from_ints(one_body_integrals: np.ndarray,
                            two_body_integrals: np.ndarray = None) -> FermionicOp:
    """
    Builds a fermionic operator based on 1- and/or 2-body integrals.
    This method requires the integrals stored in the '*chemist*' notation
             h2(i,j,k,l) --> adag_i adag_k a_l a_j
    and the integral values are used for the coefficients of the second-quantized
    Hamiltonian that is built. The integrals input here should be in block spin
    format and also have indexes reordered as follows 'ijkl->ljik'
    There is another popular notation, the '*physicist*' notation
             h2(i,j,k,l) --> adag_i adag_j a_k a_l
    If you are using the '*physicist*' notation, you need to convert it to
    the '*chemist*' notation. E.g. h2=numpy.einsum('ikmj->ijkm', h2)
    The :class:`~qiskit_nature.QMolecule` class has
    :attr:`~qiskit_nature.QMolecule.one_body_integrals` and
    :attr:`~qiskit_nature.QMolecule.two_body_integrals` properties that can be
    directly supplied to the `h1` and `h2` parameters here respectively.

    Args:
        one_body_integrals (numpy.ndarray): One-body integrals stored in the chemist notation.
        two_body_integrals (numpy.ndarray): Two-body integrals stored in the chemist notation.

    Returns:
        FermionicOp: FermionicOp built from 1- and/or 2-body integrals.
    """

    fermionic_op = _build_fermionic_op(one_body_integrals, two_body_integrals)
    fermionic_op = fermionic_op.reduce()

    return fermionic_op


def _build_fermionic_op(one_body_integrals: np.ndarray,
                        two_body_integrals: np.ndarray) -> FermionicOp:
    one_body_base_ops = _create_one_body_base_ops(one_body_integrals)
    two_body_base_ops = _create_two_body_base_ops(
        two_body_integrals) if two_body_integrals is not None else []
    base_ops = one_body_base_ops + two_body_base_ops

    fermionic_op = FermionicOp('I' * len(one_body_integrals))
    for base_op in base_ops:
        fermionic_op += base_op
    return fermionic_op


def _create_one_body_base_ops(one_body_integrals: np.ndarray):
    return _create_base_ops(one_body_integrals, 2, _calc_coeffs_with_ops_one_body)


def _create_two_body_base_ops(two_body_integrals: np.ndarray):
    return _create_base_ops(two_body_integrals, 4, _calc_coeffs_with_ops_two_body)


def _create_base_ops(integrals: np.ndarray, repeat_num: int, calc_coeffs_with_ops):
    base_ops_list = []
    integrals_length = len(integrals)
    for idx in itertools.product(range(integrals_length), repeat=repeat_num):
        coeff = integrals[idx]
        if not coeff:
            continue
        coeffs_with_ops = calc_coeffs_with_ops(idx)
        base_op = _create_base_op_from_labels(coeff, integrals_length, coeffs_with_ops)
        base_ops_list.append(base_op)
    return base_ops_list


def _calc_coeffs_with_ops_one_body(idx):
    return [(idx[0], '+'), (idx[1], '-')]


def _calc_coeffs_with_ops_two_body(idx):
    return [(idx[0], '+'), (idx[2], '+'), (idx[3], '-'), (idx[1], '-')]


def _create_base_op_from_labels(coeff, length: int, coeffs_with_ops):
    label = ['I'] * length
    base_op = coeff * FermionicOp(''.join(label))
    for i, op in coeffs_with_ops:
        label_i = label.copy()
        label_i[i] = op
        base_op @= FermionicOp(''.join(label_i))
    return base_op
