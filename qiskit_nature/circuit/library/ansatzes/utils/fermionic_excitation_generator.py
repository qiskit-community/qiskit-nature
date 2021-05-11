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
This method is used by the :class:`~.UCC` Ansatz in order to construct its excitation operators. It
must be called for each type of excitation (singles, doubles, etc.) that is to be considered in the
Ansatz.
Some keyword arguments are allowed through which the excitations can be filtered to fit the user's
needs. `alpha_spin` and `beta_spin` are boolean flags which can be set to `False` in order to
disable the inclusion of alpha-spin or beta-spin excitation, respectively.
`max_spin_excitation` takes an integer value which defines the maximum number of excitations that
can occur within the same spin. Thus, setting `max_spin_excitation=1` and `num_excitations=2` yields
only those double excitations which do not excite the same spin species twice.
"""

from typing import Iterator, List, Tuple, Optional

import itertools
import logging

logger = logging.getLogger(__name__)


def generate_fermionic_excitations(
    num_excitations: int,
    num_spin_orbitals: int,
    num_particles: Tuple[int, int],
    alpha_spin: bool = True,
    beta_spin: bool = True,
    max_spin_excitation: Optional[int] = None,
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

    Returns:
        The list of excitations encoded as tuples of tuples. Each tuple in the list is a pair. The
        first tuple contains the occupied spin orbital indices whereas the second one contains the
        indices of the unoccupied spin orbitals.
    """
    alpha_excitations: List[Tuple[int, int]] = []
    if alpha_spin:
        # generate alpha-spin orbital indices for occupied and unoccupied ones
        alpha_occ = list(range(num_particles[0]))
        alpha_unocc = list(range(num_particles[0], num_spin_orbitals // 2))
        # the Cartesian product of these lists gives all possible single alpha-spin excitations
        alpha_excitations = list(itertools.product(alpha_occ, alpha_unocc))
        logger.debug("Generated list of single alpha excitations: %s", alpha_excitations)

    beta_excitations: List[Tuple[int, int]] = []
    if beta_spin:
        # generate beta-spin orbital indices for occupied and unoccupied ones
        beta_occ = list(range(num_spin_orbitals // 2, num_spin_orbitals // 2 + num_particles[1]))
        beta_unocc = list(range(num_spin_orbitals // 2 + num_particles[1], num_spin_orbitals))
        # the Cartesian product of these lists gives all possible single beta-spin excitations
        beta_excitations = list(itertools.product(beta_occ, beta_unocc))
        logger.debug("Generated list of single beta excitations: %s", beta_excitations)

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

    excitations: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = list()
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
