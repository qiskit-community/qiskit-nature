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


def build_ferm_op_from_ints(one_body_integrals, two_body_integrals=None) -> FermionicOp:
    """
    Builds a fermionic operator based on 1- and/or 2-body integrals.

    Args:
        one_body_integrals (numpy.ndarray): One-body integrals.
        two_body_integrals (numpy.ndarray): Two-body integrals.

    Returns:
        FermionicOp: FermionicOp built from 1- and/or 2-body integrals.
    """

    fermionic_op = FermionicOp('I' * len(one_body_integrals))
    fermionic_op = _fill_ferm_op_one_body_ints(fermionic_op, one_body_integrals)
    if two_body_integrals is not None:
        fermionic_op = _fill_ferm_op_two_body_ints(fermionic_op, two_body_integrals)

    fermionic_op = fermionic_op.reduce()

    return fermionic_op


def _fill_ferm_op_one_body_ints(fermionic_op: FermionicOp, one_body_integrals):
    for idx in itertools.product(range(len(one_body_integrals)), repeat=2):
        coeff = one_body_integrals[idx]
        if not coeff:
            continue
        label = ['I'] * len(one_body_integrals)
        base_op = coeff * FermionicOp(''.join(label))
        for i, op in [(idx[0], '+'), (idx[1], '-')]:
            label_i = label.copy()
            label_i[i] = op
            base_op @= FermionicOp(''.join(label_i))
        fermionic_op += base_op
    return fermionic_op


def _fill_ferm_op_two_body_ints(fermionic_op: FermionicOp, two_body_integrals):
    for idx in itertools.product(range(len(two_body_integrals)), repeat=4):
        coeff = two_body_integrals[idx]
        if not coeff:
            continue
        label = ['I'] * len(two_body_integrals)
        base_op = coeff * FermionicOp(''.join(label))
        for i, op in [(idx[0], '+'), (idx[2], '+'), (idx[3], '-'), (idx[1], '-')]:
            label_i = label.copy()
            label_i[i] = op
            base_op @= FermionicOp(''.join(label_i))
        fermionic_op += base_op
    return fermionic_op
