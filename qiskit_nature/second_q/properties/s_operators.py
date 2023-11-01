# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Generator functions for various spin-related operators."""

from __future__ import annotations

from qiskit_nature.second_q.operators import FermionicOp


def s_plus_operator(num_spatial_orbitals: int) -> FermionicOp:
    r"""Constructs the $S^+$ operator.

    The $S^+$ operator is defined as:

    .. math::

       S^+ = \sum_{i=1}^{n} \hat{a}_{i}^{\dagger} \hat{a}_{i+n}

    Here, $n$ denotes the index of the *spatial* orbital. Since Qiskit Nature employs the blocked
    spin-ordering convention, the creation operator above is applied to the :math:`\alpha`-spin
    orbital and the annihilation operator is applied to the corresponding :math:`\beta`-spin
    orbital.

    Args:
        num_spatial_orbitals: the size of the operator which to generate.

    Returns:
        The $S^+$ operator of the requested size.
    """
    op = FermionicOp(
        {f"+_{orb} -_{orb + num_spatial_orbitals}": 1.0 for orb in range(num_spatial_orbitals)}
    )
    return op


def s_minus_operator(num_spatial_orbitals: int) -> FermionicOp:
    r"""Constructs the $S^-$ operator.

    The $S^-$ operator is defined as:

    .. math::

       S^- = \sum_{i=1}^{n} \hat{a}_{i+n}^{\dagger} \hat{a}_{i}

    Here, $n$ denotes the index of the *spatial* orbital. Since Qiskit Nature employs the blocked
    spin-ordering convention, the creation operator above is applied to the :math:`\beta`-spin
    orbital and the annihilation operator is applied to the corresponding :math:`\alpha`-spin
    orbital.

    Args:
        num_spatial_orbitals: the size of the operator which to generate.

    Returns:
        The $S^-$ operator of the requested size.
    """
    op = FermionicOp(
        {f"+_{orb + num_spatial_orbitals} -_{orb}": 1.0 for orb in range(num_spatial_orbitals)}
    )
    return op


def s_x_operator(num_spatial_orbitals: int) -> FermionicOp:
    r"""Constructs the $S^x$ operator.

    The $S^x$ operator is defined as:

    .. math::

       S^x = \frac{1}{2} \left(S^+ + S^-\right)

    Args:
        num_spatial_orbitals: the size of the operator which to generate.

    Returns:
        The $S^x$ operator of the requested size.
    """
    op = FermionicOp(
        {
            f"+_{orb} -_{(orb + num_spatial_orbitals) % (2*num_spatial_orbitals)}": 0.5
            for orb in range(2 * num_spatial_orbitals)
        }
    )
    return op


def s_y_operator(num_spatial_orbitals: int) -> FermionicOp:
    r"""Constructs the $S^y$ operator.

    The $S^y$ operator is defined as:

    .. math::

       S^y = -\frac{i}{2} \left(S^+ - S^-\right)

    Args:
        num_spatial_orbitals: the size of the operator which to generate.

    Returns:
        The $S^y$ operator of the requested size.
    """
    op = FermionicOp(
        {
            f"+_{orb} -_{(orb + num_spatial_orbitals) % (2*num_spatial_orbitals)}": 0.5j
            * (-1.0) ** (orb < num_spatial_orbitals)
            for orb in range(2 * num_spatial_orbitals)
        }
    )
    return op


def s_z_operator(num_spatial_orbitals: int) -> FermionicOp:
    r"""Constructs the $S^z$ operator.

    The $S^z$ operator is defined as:

    .. math::

       S^z = \frac{1}{2} \sum_{i=1}^{n} \left(
        \hat{a}_{i}^{\dagger}\hat{a}_{i} - \hat{a}_{i+n}^{\dagger}\hat{a}_{i+n}
       \right)

    Here, $n$ denotes the index of the *spatial* orbital. Since Qiskit Nature employs the blocked
    spin-ordering convention, this means that the above corresponds to evaluating the number
    operator (:math:`\hat{a}^{\dagger}\hat{a}`) once on the :math:`\alpha`-spin orbital and once on
    the :math:`\beta`-spin orbital and taking their difference.

    Args:
        num_spatial_orbitals: the size of the operator which to generate.

    Returns:
        The $S^z$ operator of the requested size.
    """
    op = FermionicOp(
        {
            f"+_{orb} -_{orb}": 0.5 * (-1.0) ** (orb >= num_spatial_orbitals)
            for orb in range(2 * num_spatial_orbitals)
        }
    )
    return op
