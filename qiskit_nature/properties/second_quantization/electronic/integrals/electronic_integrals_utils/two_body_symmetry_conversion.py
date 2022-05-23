# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions to detect and transform the index-ordering convention of two-body integrals"""

from enum import Enum
import numpy

from qiskit_nature import QiskitNatureError


def to_chem(two_body_tensor):
    """
    Convert the rank-four tensor `two_body_tensor` representing two-body integrals from physicists'
    , or intermediate, index order to chemists' index order: i,j,k,l -> i,l,j,k
    """
    index_order = find_index_order(two_body_tensor)
    if index_order == IndexType.CHEM:
        return two_body_tensor
    if index_order == IndexType.PHYS:
        chem_tensor = _phys_to_chem(two_body_tensor)
        return chem_tensor
    if index_order == IndexType.INT:
        chem_tensor = _chem_to_phys(two_body_tensor)
        return chem_tensor
    elif index_order == IndexType.UNKNOWN:
        raise QiskitNatureError(
            """
            Unknown index order type, input tensor must be chemists', physicists',
            or intermediate index order
            """
        )


def to_phys(two_body_tensor):
    """
    Convert the rank-four tensor `two_body_tensor` representing two-body integrals from chemists'
    , or intermediate, index order to physicists' index order: i,j,k,l -> i,l,j,k
    """
    index_order = find_index_order(two_body_tensor)
    if index_order == IndexType.PHYS:
        return two_body_tensor
    if index_order == IndexType.CHEM:
        phys_tensor = _chem_to_phys(two_body_tensor)
        return phys_tensor
    if index_order == IndexType.INT:
        phys_tensor = _phys_to_chem(two_body_tensor)
        return phys_tensor
    elif index_order == IndexType.UNKNOWN:
        raise QiskitNatureError(
            """
            Unknown index order type, input tensor must be chemists', physicists',
            or intermediate index order
            """
        )


def _phys_to_chem(two_body_tensor):
    """
    Convert the rank-four tensor `two_body_tensor` representing two-body integrals from physicists'
    index order to chemists' index order: i,j,k,l -> i,l,j,k

    See `_chem_to_phys`, `_check_two_body_symmetries`.
    """
    permuted_tensor = numpy.einsum("ijkl->iljk", two_body_tensor)
    return permuted_tensor


def _chem_to_phys(two_body_tensor):
    """
    Convert the rank-four tensor `two_body_tensor` representing two-body integrals from chemists'
    index order to physicists' index order: i,j,k,l -> i,k,l,j

    See `phys_to_chem`, `check_two_body_symmetries`.

    Note:
    Denote `chem_to_phys` by `g` and `phys_to_chem` by `h`. The elements `g`, `h`, `I` form
    a group with `gh = hg = I`, `g^2=h`, and `h^2=g`.
    """
    permuted_tensor = numpy.einsum("ijkl->iklj", two_body_tensor)
    return permuted_tensor


def _check_two_body_symmetry(tensor, permutation):
    """
    Return `True` if `tensor` passes symmetry test number `test_number`. Otherwise,
    return `False`.
    """
    permuted_tensor = numpy.einsum(permutation, tensor)
    return numpy.allclose(tensor, permuted_tensor)


def _check_two_body_symmetries(two_body_tensor, chemist=True):
    """
    Return `True` if the rank-4 tensor `two_body_tensor` has the required symmetries for coefficients
    of the two-electron terms.  If `chemist` is `True`, assume the input is in chemists' order,
    otherwise in physicists' order.

    If `two_body_tensor` is a correct tensor of indices, with the correct index order, it must pass the
    tests. If `two_body_tensor` is a correct tensor of indices, but the flag `chemist` is incorrect,
    it will fail the tests, unless the tensor has accidental symmetries.
    This test may be used with care to discriminate between the orderings.

    References: HJO Molecular Electronic-Structure Theory (1.4.17), (1.4.38)

    See `_phys_to_chem`, `_chem_to_phys`.
    """
    if not chemist:
        two_body_tensor = _phys_to_chem(two_body_tensor)
    for permutation in ChemIndexPermutations:
        if not _check_two_body_symmetry(two_body_tensor, permutation.value):
            return False
    return True


def find_index_order(two_body_tensor):
    """
    Return the index-order convention of rank-four `two_body_tensor`.

    The index convention is determined by checking symmetries of the tensor.
    If the indexing convention can be determined, then one of `:chemist`,
    `:physicist`, or `:intermediate` is returned. The `:intermediate` indexing
    may be obtained by applying `chem_to_phys` to the physicists' convention or
    `phys_to_chem` to the chemists' convention. If the tests for each of these
    conventions fail, then `:unknown` is returned.

    See also: `_chem_to_phys`, `_phys_to_chem`.

    Note:
    The first of `:chemist`, `:physicist`, and `:intermediate`, in that order, to pass the tests
    is returned. If `two_body_tensor` has accidental symmetries, it may in fact satisfy more
    than one set of symmetry tests. For example, if all elements have the same value, then the
    symmetries for all three index orders are satisfied.
    """
    if _check_two_body_symmetries(two_body_tensor):
        return IndexType.CHEM
    permuted_tensor = _phys_to_chem(two_body_tensor)
    if _check_two_body_symmetries(permuted_tensor):
        return IndexType.PHYS
    permuted_tensor = _phys_to_chem(permuted_tensor)
    if _check_two_body_symmetries(permuted_tensor):
        return IndexType.INT
    else:
        return IndexType.UNKNOWN


class ChemIndexPermutations(Enum):
    """This ``Enum`` defines the permutation symmetries satisfied by a rank-4 tensor of real
    two-body integrals in chemists' index order, naming each permutation in order of appearance
    in Molecular Electronic Structure Theory by Helgaker, Jørgensen, Olsen (HJO)."""

    PERM_1 = "pqrs->rspq"  # HJO (1.4.17)
    PERM_2_AB = "pqrs->qprs"  # HJO (1.4.38)
    PERM_2_AC = "pqrs->pqsr"  # HJO (1.4.38)
    PERM_2_AD = "pqrs->qpsr"  # HJO (1.4.38)
    PERM_3 = "pqrs->rsqp"  # PERM_2_AB and PERM_1
    PERM_4 = "pqrs->srpq"  # PERM_2_AC and PERM_1
    PERM_5 = "pqrs->srqp"  # PERM_2_AD and PERM_1


class IndexType(Enum):
    """This ``Enum`` names the different permutation index orders that could be encountered."""

    CHEM = "chemist"
    PHYS = "physicist"
    INT = "intermediate"
    UNKNOWN = "unknown"
