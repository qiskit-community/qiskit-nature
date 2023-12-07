# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Commutator Utilities (:mod:`qiskit_nature.second_q.operators.commutators`)
==========================================================================

.. currentmodule:: qiskit_nature.second_q.operators.commutators

Utility functions to compute commutators of
:class:`qiskit_nature.second_q.operators.SparseLabelOp` instances.

.. autosummary::
   :toctree: ../stubs/

   commutator
   anti_commutator
   double_commutator

"""

from .sparse_label_op import SparseLabelOp


def commutator(op_a: SparseLabelOp, op_b: SparseLabelOp) -> SparseLabelOp:
    r"""Compute commutator of `op_a` and `op_b`.

    .. math::

        AB - BA.

    Args:
        op_a: Operator A.
        op_b: Operator B.

    Returns:
        The computed commutator. If available for your kind of operator, you may want to
        ``normal_order()`` it.
    """
    return (op_a @ op_b - op_b @ op_a).simplify(atol=0)


def anti_commutator(op_a: SparseLabelOp, op_b: SparseLabelOp) -> SparseLabelOp:
    r"""Compute anti-commutator of `op_a` and `op_b`.

    .. math::
        AB + BA.

    Args:
        op_a: Operator A.
        op_b: Operator B.

    Returns:
        The computed anti--commutator. If available for your kind of operator, you may want to
        ``normal_order()`` it.
    """
    return (op_a @ op_b + op_b @ op_a).simplify(atol=0)


def double_commutator(
    op_a: SparseLabelOp,
    op_b: SparseLabelOp,
    op_c: SparseLabelOp,
    sign: bool = False,
) -> SparseLabelOp:
    r"""Compute symmetric double commutator of `op_a`, `op_b` and `op_c`.

    See also Equation (13.6.18) in [1].
    If `sign` is `False`, it returns

    .. math::
         [[A, B], C]/2 + [A, [B, C]]/2
         = (2ABC + 2CBA - BAC - CAB - ACB - BCA)/2.

    If `sign` is `True`, it returns

    .. math::
         \lbrace[A, B], C\rbrace/2 + \lbrace A, [B, C]\rbrace/2
         = (2ABC - 2CBA - BAC + CAB - ACB + BCA)/2.

    Args:
        op_a: Operator A.
        op_b: Operator B.
        op_c: Operator C.
        sign: False anti-commutes, True commutes.

    Returns:
        The computed double commutator.

    References:
        [1]: R. McWeeny.
            Methods of Molecular Quantum Mechanics.
            2nd Edition, Academic Press, 1992.
            ISBN 0-12-486552-6.
    """
    sign_num = 1 if sign else -1

    op_ab = op_a @ op_b
    op_ba = op_b @ op_a
    op_ac = op_a @ op_c
    op_ca = op_c @ op_a

    op_abc = op_ab @ op_c
    op_cba = op_c @ op_ba
    op_bac = op_ba @ op_c
    op_cab = op_c @ op_ab
    op_acb = op_ac @ op_b
    op_bca = op_b @ op_ca

    res = (
        op_abc
        - sign_num * op_cba
        + 0.5 * (-op_bac + sign_num * op_cab - op_acb + sign_num * op_bca)
    )

    return res.simplify(atol=0)
