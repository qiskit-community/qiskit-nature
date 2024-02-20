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

r"""Generator functions for various spin-related operators.

When dealing with non-orthonormal orbitals, you need to make sure that you include the `overlap`
matrices when using the methods below. This ensures that the operators can resolve any spin
contamination that may be present in your orbitals.

The overlap matrices that you provide have to be computed in the same basis in which the spin
operator is encoded. If you are working in the molecular orbital (MO) basis, the overlap can be
easily constructed starting from the atomic orbital (AO) overlap matrix, which can be obtained from
any standard quantum chemistry program (for example from the `get_oplp()` method in PySCF). This AO
overlap matrix can be transformed to the MO basis using the AO-to-MO transformation matrix, $C$,
according to the following equation:

.. math::

   s^{MO} = C^T s^{AO} C.

For restricted spin orbitals (i.e. :math:`C_\alpha == C_\beta`), the equation above simplifies to
the identity matrix (because the MOs will be orthonormal), in which case you can omit the `overlap`
arguments below). Otherwise, you must include the correct overlap. For example, the overlap-matrix
between the $`\alpha`$- and $`\beta`$-spin orbitals is:

.. math::

   s^{\alpha,\beta} = C_\alpha^T s^{AO} C_\beta.
"""

from __future__ import annotations

import numpy as np

from qiskit_nature.second_q.operators import FermionicOp


def s_plus_operator(num_spatial_orbitals: int, overlap: np.ndarray | None = None) -> FermionicOp:
    r"""Constructs the $S^+$ operator.

    The $S^+$ operator is defined as:

    .. math::

       S^+ = \sum_{i,j} s_{ij}^{\alpha,\beta} \hat{a}_{i}^{\dagger} \hat{a}_{j},

    where $s$ denotes the overlap-matrix between the $\alpha$- and $\beta$-spin orbitals.

    Note that for orthonormal orbitals this overlap-matrix will become the identity matrix,
    simplifying the operator above to become:

    .. math::

       S^+ = \sum_{i=1}^{n} \hat{a}_{i}^{\dagger} \hat{a}_{i+n},

    where, $n$ denotes the index of the *spatial* orbital. Since Qiskit Nature employs the blocked
    spin-ordering convention, the creation operator above is applied to the :math:`\alpha`-spin
    orbital and the annihilation operator is applied to the corresponding :math:`\beta`-spin
    orbital.

    Args:
        num_spatial_orbitals: the size of the operator which to generate.
        overlap: the overlap-matrix between the $\alpha$- and $\beta$-spin orbitals. When this is
            `None`, the overlap-matrix is assumed to be identity, resulting in the second definition
            above.

    Returns:
        The $S^+$ operator of the requested size.
    """
    if overlap is None:
        op = FermionicOp(
            {f"+_{orb} -_{orb + num_spatial_orbitals}": 1.0 for orb in range(num_spatial_orbitals)}
        )
    else:
        op = FermionicOp(
            {
                f"+_{idx[0]} -_{idx[1] + num_spatial_orbitals}": overlap[idx]
                for idx in np.ndindex(*overlap.shape)
            }
        )
    return op.simplify()


def s_minus_operator(num_spatial_orbitals: int, overlap: np.ndarray | None = None) -> FermionicOp:
    r"""Constructs the $S^-$ operator.

    The $S^-$ operator is defined as:

    .. math::

       S^- = \sum_{i,j} s_{ij}^{\beta,\alpha} \hat{a}_{i}^{\dagger} \hat{a}_{j},

    where $s$ denotes the overlap-matrix between the $\beta$- and $\alpha$-spin orbitals.

    .. note::

       The `overlap` input to this method is related to the input of the other methods
       (:meth:`s_plus_operator`, :meth:`s_x_operator`, and :meth:`s_y_operator`) by its transpose,
       because the following relation holds:

       .. math::

          s_{ij}^{\beta,\alpha} = \left(s_{ij}^{\alpha,\beta}\right)^T.

    Note that for orthonormal orbitals this overlap-matrix will become the identity matrix,
    simplifying the operator above to become:

       S^- = \sum_{i=1}^{n} \hat{a}_{i+n}^{\dagger} \hat{a}_{i}

    where, $n$ denotes the index of the *spatial* orbital. Since Qiskit Nature employs the blocked
    spin-ordering convention, the creation operator above is applied to the :math:`\beta`-spin
    orbital and the annihilation operator is applied to the corresponding :math:`\alpha`-spin
    orbital.

    Args:
        num_spatial_orbitals: the size of the operator which to generate.
        overlap: the overlap-matrix between the $\beta$- and $\alpha$-spin orbitals. When this is
            `None`, the overlap-matrix is assumed to be identity, resulting in the second definition
            above.

    Returns:
        The $S^-$ operator of the requested size.
    """
    if overlap is None:
        op = FermionicOp(
            {f"+_{orb + num_spatial_orbitals} -_{orb}": 1.0 for orb in range(num_spatial_orbitals)}
        )
    else:
        op = FermionicOp(
            {
                f"+_{idx[0] + num_spatial_orbitals} -_{idx[1]}": overlap[idx]
                for idx in np.ndindex(*overlap.shape)
            }
        )
    return op.simplify()


def s_x_operator(num_spatial_orbitals: int, overlap: np.ndarray | None = None) -> FermionicOp:
    r"""Constructs the $S^x$ operator.

    The $S^x$ operator is defined as:

    .. math::

       S^x = \frac{1}{2} \left(S^+ + S^-\right)

    Args:
        num_spatial_orbitals: the size of the operator which to generate.
        overlap: the overlap-matrix between the $\alpha$- and $\beta$-spin orbitals. When this is
            `None`, the overlap-matrix is assumed to be identity.

    Returns:
        The $S^x$ operator of the requested size.
    """
    if overlap is None:
        op = FermionicOp(
            {
                f"+_{orb} -_{(orb + num_spatial_orbitals) % (2*num_spatial_orbitals)}": 0.5
                for orb in range(2 * num_spatial_orbitals)
            }
        )
    else:
        op = 0.5 * (
            s_plus_operator(num_spatial_orbitals, overlap)
            + s_minus_operator(num_spatial_orbitals, overlap.T)
        )
    return op


def s_y_operator(num_spatial_orbitals: int, overlap: np.ndarray | None = None) -> FermionicOp:
    r"""Constructs the $S^y$ operator.

    The $S^y$ operator is defined as:

    .. math::

       S^y = -\frac{i}{2} \left(S^+ - S^-\right)

    Args:
        num_spatial_orbitals: the size of the operator which to generate.
        overlap: the overlap-matrix between the $\alpha$- and $\beta$-spin orbitals. When this is
            `None`, the overlap-matrix is assumed to be identity.

    Returns:
        The $S^y$ operator of the requested size.
    """
    if overlap is None:
        op = FermionicOp(
            {
                f"+_{orb} -_{(orb + num_spatial_orbitals) % (2*num_spatial_orbitals)}": 0.5j
                * (-1.0) ** (orb < num_spatial_orbitals)
                for orb in range(2 * num_spatial_orbitals)
            }
        )
    else:
        op = -0.5j * (
            s_plus_operator(num_spatial_orbitals, overlap)
            - s_minus_operator(num_spatial_orbitals, overlap.T)
        )
    return op


def s_z_operator(num_spatial_orbitals: int) -> FermionicOp:
    r"""Constructs the $S^z$ operator.

    The $S^z$ operator is defined as:

    .. math::

       S^z = \frac{1}{2} \sum_{i=1}^{n} \left(
        \hat{a}_{i}^{\dagger}\hat{a}_{i} - \hat{a}_{i+n}^{\dagger}\hat{a}_{i+n}
       \right),

    where, $n$ denotes the index of the *spatial* orbital. Since Qiskit Nature employs the blocked
    spin-ordering convention, this means that the above corresponds to evaluating the number
    operator (:math:`\hat{a}^{\dagger}\hat{a}`) once on the :math:`\alpha`-spin orbital and once on
    the :math:`\beta`-spin orbital and taking their difference.

    .. note::

       Contrary to the other methods in this module, this one does not require the inclusion of an
       overlap-matrix for non-orthonormal orbitals, because it does not mix the $\alpha$- and
       $\beta$-spin contributions.

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
