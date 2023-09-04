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
Tensor Ordering Utilities (:mod:`qiskit_nature.second_q.operators.tensor_ordering`)
===================================================================================

.. currentmodule:: qiskit_nature.second_q.operators.tensor_ordering

Utility functions to detect and transform the index-ordering convention of two-body integrals

.. autosummary::
   :toctree: ../stubs/

   to_chemist_ordering
   to_physicist_ordering
   find_index_order
   IndexType

"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from qiskit_nature import QiskitNatureError
import qiskit_nature.optionals as _optionals

if TYPE_CHECKING:
    from .symmetric_two_body import SymmetricTwoBodyIntegrals

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray
else:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


def to_chemist_ordering(
    two_body_tensor: np.ndarray | SparseArray,
    *,
    index_order: IndexType | None = None,
) -> np.ndarray | SparseArray:
    """Convert a two-body tensor to chemists' index order.

    Coverts the rank-four tensor two-body tensor representing two-body integrals from
    physicists', or intermediate, index order to chemists' index order:
    ``i,j,k,l -> i,l,j,k``

    Args:
        two_body_tensor: the rank-four tensor to be converted.
        index_order: when supplied this will hard-code the ``IndexType`` value. If ``None`` (the
            default), the ``index_order`` will be determined automatically based on the symmetries of
            the ``two_body_tensor``.

    Returns:
        The same rank-four tensor, now in chemists' index order.

    Raises:
        QiskitNatureError: when an unknown index type is encountered.
    """
    if index_order is None:
        index_order = find_index_order(two_body_tensor)
    if index_order == IndexType.CHEMIST:
        return two_body_tensor
    if index_order == IndexType.PHYSICIST:
        chem_tensor = _phys_to_chem(two_body_tensor)
        return chem_tensor
    if index_order == IndexType.INTERMEDIATE:
        chem_tensor = _chem_to_phys(two_body_tensor)
        return chem_tensor
    else:
        raise QiskitNatureError(
            """
            Unknown index order type, input tensor must be chemists', physicists',
            or intermediate index order
            """
        )


def to_physicist_ordering(
    two_body_tensor: np.ndarray | SparseArray,
    *,
    index_order: IndexType | None = None,
) -> np.ndarray | SparseArray:
    """Convert a two-body tensor to physicists' index order.

    Converts the rank-four tensor two-body tensor representing two-body integrals from
    chemists', or intermediate, index order to physicists' index order: ``i,j,k,l -> i,l,j,k``

    Args:
        two_body_tensor: the rank-four tensor to be converted.
        index_order: when supplied this will hard-code the ``IndexType`` value. If ``None`` (the
            default), the ``index_order`` will be determined automatically based on the symmetries of
            the ``two_body_tensor``.

    Returns:
        The same rank-four tensor, now in physicists' index order.

    Raises:
        QiskitNatureError: when an unknown index type is encountered.
    """
    if index_order is None:
        index_order = find_index_order(two_body_tensor)
    if index_order == IndexType.PHYSICIST:
        return two_body_tensor
    if index_order == IndexType.CHEMIST:
        phys_tensor = _chem_to_phys(two_body_tensor)
        return phys_tensor
    if index_order == IndexType.INTERMEDIATE:
        phys_tensor = _phys_to_chem(two_body_tensor)
        return phys_tensor
    else:
        raise QiskitNatureError(
            """
            Unknown index order type, input tensor must be chemists', physicists',
            or intermediate index order
            """
        )


def _phys_to_chem(two_body_tensor: np.ndarray | SparseArray) -> np.ndarray | SparseArray:
    """Convert the rank-four tensor `two_body_tensor` representing two-body integrals from
    physicists' index order to chemists' index order: i,j,k,l -> i,l,j,k

    See also `_chem_to_phys`, `_check_two_body_symmetries`.

    .. note::
      Denote `_chem_to_phys` by `g` and `_phys_to_chem` by `h`. The elements `g`, `h`, `I` form
      a group with `gh = hg = I`, `g^2=h`, and `h^2=g`.

    Args:
        two_body_tensor: the rank-four tensor in physicists' to be converted.

    Returns:
        The same rank-four tensor, now in chemists' index order.
    """
    permuted_tensor = np.moveaxis(two_body_tensor, (1, 2), (2, 3))
    return permuted_tensor


def _chem_to_phys(two_body_tensor: np.ndarray | SparseArray) -> np.ndarray | SparseArray:
    """Convert the rank-four tensor `two_body_tensor` representing two-body integrals from chemists'
    index order to physicists' index order: i,j,k,l -> i,k,l,j

    See also `_phys_to_chem`, `_check_two_body_symmetries`.

    .. note::
      Denote `_chem_to_phys` by `g` and `_phys_to_chem` by `h`. The elements `g`, `h`, `I` form
      a group with `gh = hg = I`, `g^2=h`, and `h^2=g`.

    Args:
        two_body_tensor: the rank-four tensor in chemists' to be converted.

    Returns:
        The same rank-four tensor, now in physicists' index order.
    """
    permuted_tensor = np.moveaxis(two_body_tensor, (1,), (3,))
    return permuted_tensor


def _check_two_body_symmetry(
    two_body_tensor: np.ndarray | SparseArray,
    permutation: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Return whether the provided tensor remains identical under the provided permutation.

    Args:
        two_body_tensor: the tensor to test.
        permutation: the source and destination indices of the axis permutations.
        rtol: the relative tolerance used during the comparison.
        atol: the absolute tolerance used during the comparison.

    Returns:
        Whether the tensor remains unchanged under the applied permutation.
    """
    permuted_tensor = np.moveaxis(two_body_tensor, permutation[0], permutation[1])

    if isinstance(two_body_tensor, SparseArray):
        return np.allclose(
            two_body_tensor.data, permuted_tensor.data, rtol=rtol, atol=atol
        ) and np.array_equal(
            two_body_tensor.coords, permuted_tensor.coords  # type: ignore[attr-defined]
        )

    return np.allclose(two_body_tensor, permuted_tensor, rtol=rtol, atol=atol)


def _check_two_body_symmetries(
    two_body_tensor: np.ndarray | SparseArray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Return whether a tensor has the required symmetries to represent two-electron terms.

    Return `True` if the rank-4 tensor `two_body_tensor` has the required symmetries for
    coefficients of the two-electron terms.

    If `two_body_tensor` is a correct tensor of indices, with the correct index order, it must pass
    the tests. If `two_body_tensor` is a correct tensor of indices, but the flag `chemist` is
    incorrect, it will fail the tests, unless the tensor has accidental symmetries. This test may be
    used with care to discriminate between the orderings.

    References: HJO Molecular Electronic-Structure Theory (1.4.17), (1.4.38)

    See also `_phys_to_chem`, `_chem_to_phys`.

    Args:
        two_body_tensor: the tensor to test.
        rtol: the relative tolerance used during the comparison.
        atol: the absolute tolerance used during the comparison.

    Returns:
        Whether the tensor has the required symmetries to represent two-electron terms.
    """
    for permutation in _ChemIndexPermutations:
        if not _check_two_body_symmetry(two_body_tensor, permutation.value, rtol=rtol, atol=atol):
            return False
    return True


def find_index_order(
    two_body_tensor: np.ndarray | SparseArray | SymmetricTwoBodyIntegrals,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> IndexType:
    """Return the index-order convention of the provided rank-four tensor.

    The index convention is determined by checking symmetries of the tensor.
    If the indexing convention can be determined, then one of :class:`IndexType.CHEMIST`,
    :class:`IndexType.PHYSICIST`, or :class:`IndexType.INTERMEDIATE` is returned. The
    :class:`IndexType.INTERMEDIATE` indexing may be obtained by applying :meth:`_chem_to_phys` to
    the physicists' convention or :meth:`_phys_to_chem` to the chemists' convention. If the tests
    for each of these conventions fail, then :class:`IndexType.UNKNOWN` is returned.

    .. note::
      The first of :class:`IndexType.CHEMIST`, :class:`IndexType.PHYSICIST`, and
      :class:`IndexType.INTERMEDIATE`, in that order, to pass the tests is returned. If
      ``two_body_tensor`` has accidental symmetries, it may in fact satisfy more than one set of
      symmetry tests. For example, if all elements have the same value, then the symmetries for all
      three index orders are satisfied.

    Args:
        two_body_tensor: the rank-four tensor whose index order to determine.
        rtol: the relative tolerance used during the comparison.
        atol: the absolute tolerance used during the comparison.

    Returns:
        The index order of the provided rank-four tensor.
    """
    from .symmetric_two_body import SymmetricTwoBodyIntegrals

    if isinstance(two_body_tensor, SymmetricTwoBodyIntegrals):
        return IndexType.CHEMIST

    if _check_two_body_symmetries(two_body_tensor, rtol=rtol, atol=atol):
        return IndexType.CHEMIST
    permuted_tensor = _phys_to_chem(two_body_tensor)
    if _check_two_body_symmetries(permuted_tensor, rtol=rtol, atol=atol):
        return IndexType.PHYSICIST
    permuted_tensor = _phys_to_chem(permuted_tensor)
    if _check_two_body_symmetries(permuted_tensor, rtol=rtol, atol=atol):
        return IndexType.INTERMEDIATE
    else:
        return IndexType.UNKNOWN


class _ChemIndexPermutations(Enum):
    """This ``Enum`` defines the permutation symmetries satisfied by a rank-4 tensor of real
    two-body integrals in chemists' index order, naming each permutation in order of appearance
    in Molecular Electronic Structure Theory by Helgaker, Jørgensen, Olsen (HJO)."""

    PERM_1 = ((0, 1), (2, 3))  # HJO (1.4.17)
    PERM_2_AB = ((0,), (1,))  # HJO (1.4.38)
    PERM_2_AC = ((2,), (3,))  # HJO (1.4.38)
    PERM_2_AD = ((0, 2), (1, 3))  # HJO (1.4.38)
    PERM_3 = ((0, 1), (3, 2))  # PERM_2_AB and PERM_1
    PERM_4 = ((0, 1, 2), (2, 3, 1))  # PERM_2_AC and PERM_1
    PERM_5 = ((0, 1, 2), (3, 2, 1))  # PERM_2_AD and PERM_1


class IndexType(Enum):
    """This ``Enum`` names the different permutation index orders that could be encountered."""

    CHEMIST = "chemist"
    PHYSICIST = "physicist"
    INTERMEDIATE = "intermediate"
    UNKNOWN = "unknown"
