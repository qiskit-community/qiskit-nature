# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
These utility methods are used by the :class:`~.UCC` Ansatz in order to construct its excitation
operators.
"""

from __future__ import annotations

from typing import Iterator

import itertools
import logging

logger = logging.getLogger(__name__)


def get_alpha_excitations(
    num_spatial_orbitals: int,
    num_alpha: int,
    *,
    generalized: bool = False,
) -> list[tuple[int, int]]:
    """Generates all possible single alpha-electron excitations.

    This method assumes block-ordered spin-orbitals.

    Args:
        num_alpha: the number of alpha electrons.
        num_spatial_orbitals: the number of spatial-orbitals.
        generalized: boolean flag whether or not to use generalized excitations, which ignore the
            occupation of the spin orbitals. As such, the set of generalized excitations is only
            determined from the number of spin orbitals and independent from the number of alpha
            electrons.

    Returns:
        The list of excitations encoded as tuples. Each tuple is a pair. The first entry contains
        the occupied spin orbital index and the second entry the unoccupied one.
    """
    if generalized:
        return list(itertools.combinations(range(num_spatial_orbitals), 2))

    alpha_occ = range(num_alpha)
    alpha_unocc = range(num_alpha, num_spatial_orbitals)

    return list(itertools.product(alpha_occ, alpha_unocc))


def get_beta_excitations(
    num_spatial_orbitals: int,
    num_beta: int,
    *,
    generalized: bool = False,
) -> list[tuple[int, int]]:
    """Generates all possible single beta-electron excitations.

    This method assumes block-ordered spin-orbitals.

    Args:
        num_beta: the number of beta electrons.
        num_spatial_orbitals: the total number of spatial-orbitals.
        generalized: boolean flag whether or not to use generalized excitations, which ignore the
            occupation of the spin orbitals. As such, the set of generalized excitations is only
            determined from the number of spin orbitals and independent from the number of beta
            electrons.

    Returns:
        The list of excitations encoded as tuples. Each tuple is a pair. The first entry contains
        the occupied spin orbital index and the second entry the unoccupied one.
    """
    num_spin_orbitals = 2 * num_spatial_orbitals
    if generalized:
        return list(itertools.combinations(range(num_spatial_orbitals, num_spin_orbitals), 2))

    beta_index_offset = num_spatial_orbitals
    beta_occ = range(beta_index_offset, beta_index_offset + num_beta)
    beta_unocc = range(beta_index_offset + num_beta, num_spin_orbitals)

    return list(itertools.product(beta_occ, beta_unocc))


def generate_fermionic_excitations(
    num_excitations: int,
    num_spatial_orbitals: int,
    num_particles: tuple[int, int],
    *,
    alpha_spin: bool = True,
    beta_spin: bool = True,
    max_spin_excitation: int | None = None,
    generalized: bool = False,
    preserve_spin: bool = True,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    # pylint: disable=line-too-long
    """Generates all possible excitations with the given number of excitations for the specified
    number of particles distributed among the given number of spatial orbitals.

    The method must be called for each type of excitation (singles, doubles, etc.) that is to be
    considered in the Ansatz. Excitations will be produced based on an initial `Hartree-Fock`
    occupation by default unless `generalized` is set to `True`, in which case the excitations are
    only determined based on the number of spatial orbitals and are independent from
    the number of particles.

    This method assumes block-ordered spin-orbitals.

    Args:
        num_excitations: number of excitations per operator (1 means single excitations, etc.).
        num_spatial_orbitals: number of spatial-orbitals.
        num_particles: number of alpha and beta particles.
        alpha_spin: boolean flag whether to include alpha-spin excitations.
        beta_spin: boolean flag whether to include beta-spin excitations.
        max_spin_excitation: the largest number of excitations within a spin. E.g. you can set
                             this to 1 and `num_excitations` to 2 in order to obtain only
                             mixed-spin double excitations (alpha,beta) but no pure-spin double
                             excitations (alpha,alpha or beta,beta).
        generalized: boolean flag whether or not to use generalized excitations, which ignore the
            occupation of the spin orbitals. As such, the set of generalized excitations is only
            determined from the number of spin orbitals and independent from the number of
            particles.
        preserve_spin: boolean flag whether or not to preserve the particle spins.

    Returns:
        The list of excitations encoded as tuples of tuples. Each tuple in the list is a pair. The
        first tuple contains the occupied spin orbital indices whereas the second one contains the
        indices of the unoccupied spin orbitals.

    Examples:
        Generate excitations with basic inputs.

        >>> from qiskit_nature.second_q.circuit.library.ansatzes.utils.fermionic_excitation_generator import generate_fermionic_excitations
        >>> generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=3, num_particles=(1,1))
        [((0,), (1,)), ((0,), (2,)), ((3,), (4,)), ((3,), (5,))]

        Generate generalized excitations.

        >>> generate_fermionic_excitations(1, 3, (1, 1), generalized=True)
        [((0,), (1,)), ((0,), (2,)), ((1,), (2,)), ((3,), (4,)), ((3,), (5,)), ((4,), (5,))]

    """
    num_spin_orbitals = 2 * num_spatial_orbitals
    alpha_excitations: list[tuple[int, int]] = []
    beta_excitations: list[tuple[int, int]] = []

    if preserve_spin:
        if alpha_spin:
            alpha_excitations = get_alpha_excitations(
                num_spatial_orbitals, num_particles[0], generalized=generalized
            )
            logger.debug("Generated list of single alpha excitations: %s", alpha_excitations)

        if beta_spin:
            beta_excitations = get_beta_excitations(
                num_spatial_orbitals, num_particles[1], generalized=generalized
            )
            logger.debug("Generated list of single beta excitations: %s", beta_excitations)

    else:
        if generalized:
            # Combining generalized=True with preserve_spin=False results in all possible
            # excitations, regardless of both, orbital occupancy and orbital spin species.
            # This effectively amounts to all permutations of available orbitals. However,
            # this does _not_ include de-excitations, which need to be filtered!

            # First, we get the generalized alpha-spin single excitations
            single_excitations = get_alpha_excitations(
                num_spatial_orbitals, sum(num_particles), generalized=True
            )

            # We can now obtain the alpha excitations by complementing the previously generated list
            # of single excitations with the non-spin-preserving excitations.
            alpha_excitations = sorted(
                itertools.chain.from_iterable(
                    itertools.starmap(
                        lambda i, a: [(i, a), (i, a + num_spatial_orbitals)], single_excitations
                    )
                )
            )
            # The beta excitations are identical but starting from beta-spin indices
            beta_excitations = sorted(
                itertools.chain.from_iterable(
                    itertools.starmap(
                        lambda i, a: [
                            (i + num_spatial_orbitals, a),
                            (i + num_spatial_orbitals, a + num_spatial_orbitals),
                        ],
                        single_excitations,
                    )
                )
            )
        else:
            # preserve_spin=False doesn't distinguish between alpha and beta spin species. This is
            # effectively the same scenario as a single spin species for a system of double the
            # actual size up to a reordering of the orbitals. We can reuse single spin species
            # excitation generator if we reorder the orbitals afterwards. The first num_particles[0]
            # orbitals in the output are fine, but the next num_particles[1] orbitals have to be
            # reordered to start at index num_spatial_orbitals

            single_excitations = get_alpha_excitations(
                num_spin_orbitals, sum(num_particles), generalized=False
            )

            def reorder_index(index: int) -> int:
                # Alpha spins already at correct index
                if index < num_particles[0]:
                    return index
                # Cyclically permute remaining (num_spin_orbitals - num_particles[0]) orbitals to
                # get Beta spins at correct index
                else:
                    offset = num_particles[0]
                    period = num_spin_orbitals - offset
                    shift = num_spatial_orbitals - offset
                    return (index - offset + shift) % period + offset

            for (occ_idx, unocc_idx) in single_excitations:
                # we map from interleaved to blocked spin orbital indices
                reordered_occ_idx = reorder_index(occ_idx)
                reordered_unocc_idx = reorder_index(unocc_idx)
                reordered_excitation = (reordered_occ_idx, reordered_unocc_idx)

                if reordered_occ_idx < num_spatial_orbitals:
                    alpha_excitations.append(reordered_excitation)
                else:
                    beta_excitations.append(reordered_excitation)

            # NOTE: we sort the lists to ensure that non-spin flipped variants take higher precedence
            alpha_excitations = sorted(alpha_excitations)
            beta_excitations = sorted(beta_excitations)

    if not alpha_excitations and not beta_excitations:
        # nothing to do, let's return early
        return []

    # we can find the actual list of excitations by doing the following:
    #   1. combine the single alpha- and beta-spin excitations
    #   2. find all possible combinations of length `num_excitations`
    pool: Iterator[tuple[tuple[int, int], ...]] = itertools.combinations(
        alpha_excitations + beta_excitations, num_excitations
    )

    # if max_spin_excitation is set, we need to filter the pool of excitations
    if max_spin_excitation is not None:
        logger.info(
            "The maximum number of excitations within each spin species was set to %s",
            max_spin_excitation,
        )
        # first, remove all those excitations, in which more than max_spin_excitation alpha
        # excitations are performed at ones
        if alpha_excitations:  # False if empty list
            alpha_exc_set = set(alpha_excitations)
            pool = itertools.filterfalse(
                lambda exc: len(set(exc) & alpha_exc_set) > max_spin_excitation, pool
            )
        # then, do the same for beta
        if beta_excitations:  # False if empty list
            beta_exc_set = set(beta_excitations)
            pool = itertools.filterfalse(
                lambda exc: len(set(exc) & beta_exc_set) > max_spin_excitation, pool
            )

    excitations: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    visited_excitations = set()

    for exc in pool:
        # validate an excitation by asserting that all indices are unique:
        #   1. get the frozen set of indices in the excitation
        exc_set = frozenset(itertools.chain.from_iterable(exc))
        #   2. all indices must be unique (size of set equals 2 * num_excitations)
        #   3. and we also don't want to include permuted variants of identical excitations
        if len(exc_set) == num_excitations * 2 and exc_set not in visited_excitations:
            visited_excitations.add(exc_set)
            occ: tuple[int, ...]
            unocc: tuple[int, ...]
            occ, unocc = zip(*exc)
            exc_tuple = (occ, unocc)
            excitations.append(exc_tuple)
            logger.debug("Added the excitation: %s", exc_tuple)

    return excitations
