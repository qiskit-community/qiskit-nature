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
"""
These utility methods are used by the :class:`~.UCC` Ansatz in order to construct its excitation
operators. The `generate_fermionic_excitations` method must be called for each type of excitation
(singles, doubles, etc.) that is to be considered in the Ansatz.
Some keyword arguments are allowed through which the excitations can be filtered to fit the user's
needs. `alpha_spin` and `beta_spin` are boolean flags which can be set to `False` in order to
disable the inclusion of alpha-spin or beta-spin excitation, respectively.
`max_spin_excitation` takes an integer value which defines the maximum number of excitations that
can occur within the same spin. Thus, setting `max_spin_excitation=1` and `num_excitations=2` yields
only those double excitations which do not excite the same spin species twice.
`generalized` is another boolean flag which enables generalized excitations which are effectively
ignoring the spin orbital occupancies. Therefore, the excitations are only determined based on the
number of spin orbitals and are independent from the number of particles.
"""

from typing import Iterator, List, Tuple, Optional

import itertools
import logging

logger = logging.getLogger(__name__)


def get_alpha_excitations(
    num_alpha: int,
    num_spin_orbitals: int,
    generalized: bool = False,
) -> List[Tuple[int, int]]:
    """Generates all possible single alpha-electron excitations.

    This method assumes block-ordered spin-orbitals.

    Args:
        num_alpha: the number of alpha electrons.
        num_spin_orbitals: the total number of spin-orbitals (alpha + alpha spin).
        generalized: boolean flag whether or not to use generalized excitations, which ignore the
            occupation of the spin orbitals. As such, the set of generalized excitations is only
            determined from the number of spin orbitals and independent from the number of alpha
            electrons.

    Returns:
        The list of excitations encoded as tuples. Each tuple is a pair. The first entry contains
        the occupied spin orbital index and the second entry the unoccupied one.
    """
    if generalized:
        return list(itertools.combinations(range(num_spin_orbitals // 2), 2))

    alpha_occ = range(num_alpha)
    alpha_unocc = range(num_alpha, num_spin_orbitals // 2)

    return list(itertools.product(alpha_occ, alpha_unocc))


def get_beta_excitations(
    num_beta: int,
    num_spin_orbitals: int,
    generalized: bool = False,
) -> List[Tuple[int, int]]:
    """Generates all possible single beta-electron excitations.

    This method assumes block-ordered spin-orbitals.

    Args:
        num_beta: the number of beta electrons.
        num_spin_orbitals: the total number of spin-orbitals (alpha + beta spin).
        generalized: boolean flag whether or not to use generalized excitations, which ignore the
            occupation of the spin orbitals. As such, the set of generalized excitations is only
            determined from the number of spin orbitals and independent from the number of beta
            electrons.

    Returns:
        The list of excitations encoded as tuples. Each tuple is a pair. The first entry contains
        the occupied spin orbital index and the second entry the unoccupied one.
    """
    if generalized:
        return list(itertools.combinations(range(num_spin_orbitals // 2, num_spin_orbitals), 2))

    beta_index_offset = num_spin_orbitals // 2
    beta_occ = range(beta_index_offset, beta_index_offset + num_beta)
    beta_unocc = range(beta_index_offset + num_beta, num_spin_orbitals)

    return list(itertools.product(beta_occ, beta_unocc))


def generate_fermionic_excitations(
    num_excitations: int,
    num_spin_orbitals: int,
    num_particles: Tuple[int, int],
    alpha_spin: bool = True,
    beta_spin: bool = True,
    max_spin_excitation: Optional[int] = None,
    generalized: bool = False,
    preserve_spin: bool = True,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Generates all possible excitations with the given number of excitations for the specified
    number of particles distributed among the given number of spin orbitals.

    This method assumes block-ordered spin-orbitals.

    Args:
        num_excitations: number of excitations per operator (1 means single excitations, etc.).
        num_spin_orbitals: number of spin-orbitals.
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
    """
    alpha_excitations: List[Tuple[int, int]] = []
    beta_excitations: List[Tuple[int, int]] = []

    if preserve_spin:
        if alpha_spin:
            alpha_excitations = get_alpha_excitations(
                num_particles[0], num_spin_orbitals, generalized
            )
            logger.debug("Generated list of single alpha excitations: %s", alpha_excitations)

        if beta_spin:
            beta_excitations = get_beta_excitations(
                num_particles[1], num_spin_orbitals, generalized
            )
            logger.debug("Generated list of single beta excitations: %s", beta_excitations)

    else:
        # We can reuse our existing implementation for the scenario involving spin flips by
        # generating the single excitations of an _interleaved_ spin orbital system.
        # For this, we can reuse the alpha single excitation generator in a system of double the
        # actual size.
        single_excitations = get_alpha_excitations(sum(num_particles), num_spin_orbitals * 2, False)

        def interleaved2blocked(index: int, total: int) -> int:
            if index % 2 == 0:
                return index // 2

            return (index - 1 + total) // 2

        # we now split the generated single excitations into separate spin species
        for (occ_interleaved, unocc_interleaved) in single_excitations:
            # we map from interleaved to blocked spin orbital indices
            occ_blocked = interleaved2blocked(occ_interleaved, num_spin_orbitals)
            unocc_blocked = interleaved2blocked(unocc_interleaved, num_spin_orbitals)

            if occ_interleaved % 2 == 0:
                # the originally occupied orbital was of alpha-spin character
                alpha_excitations.append((occ_blocked, unocc_blocked))
            else:
                beta_excitations.append((occ_blocked, unocc_blocked))

        # NOTE: we sort the lists to ensure that non-spin flipped variants take higher precedence
        alpha_excitations = sorted(alpha_excitations)
        beta_excitations = sorted(beta_excitations)

    if not alpha_excitations and not beta_excitations:
        # nothing to do, let's return early
        return []

    # we can find the actual list of excitations by doing the following:
    #   1. combine the single alpha- and beta-spin excitations
    #   2. find all possible combinations of length `num_excitations`
    pool: Iterator[Tuple[Tuple[int, int], ...]] = itertools.combinations(
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

    excitations: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    visited_excitations = set()

    for exc in pool:
        # validate an excitation by asserting that all indices are unique:
        #   1. get the frozen set of indices in the excitation
        exc_set = frozenset(itertools.chain.from_iterable(exc))
        #   2. all indices must be unique (size of set equals 2 * num_excitations)
        #   3. and we also don't want to include permuted variants of identical excitations
        if len(exc_set) == num_excitations * 2 and exc_set not in visited_excitations:
            visited_excitations.add(exc_set)
            occ: Tuple[int, ...]
            unocc: Tuple[int, ...]
            occ, unocc = zip(*exc)
            exc_tuple = (occ, unocc)
            excitations.append(exc_tuple)
            logger.debug("Added the excitation: %s", exc_tuple)

    return excitations
