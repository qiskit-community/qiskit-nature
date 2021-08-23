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
This method is used by the :class:`~.UVCC` Ansatz in order to construct its excitation operators. It
must be called for each type of excitation (singles, doubles, etc.) that is to be considered in the
Ansatz.
"""

from typing import Any, List, Tuple

import itertools
import logging
import operator

logger = logging.getLogger(__name__)


def generate_vibration_excitations(
    num_excitations: int,
    num_modals: List[int],
) -> List[Tuple[Tuple[Any, ...], ...]]:
    """Generates all possible excitations with the given number of excitations for the specified
    number of particles distributed among the given number of spin orbitals.

    This method assumes block-ordered spin-orbitals.

    Args:
        num_excitations: number of excitations per operator (1 means single excitations, etc.).
        num_modals: the number of modals per mode.

    Returns:
        The list of excitations encoded as tuples of tuples. Each tuple in the list is a pair of
        tuples. The first tuple contains the occupied spin orbital indices whereas the second one
        contains the indices of the unoccupied spin orbitals.
    """
    partial_sum_modals = list(itertools.accumulate(num_modals, operator.add))

    # First, we construct the list of single excitations:
    single_excitations = []
    idx_sum = 0

    for accumulated_sum in partial_sum_modals:
        # the unoccupied modals in each mode are all modals but the lowest one:
        unoccupied = list(range(idx_sum + 1, accumulated_sum))
        # the single excitations for this mode are therefore simply each entry in this list, when
        # excited into it from the lowest modal of this list:
        single_excitations.extend([(idx_sum, m) for m in unoccupied])
        # and now we update the running index of the lowest modal for the next mode
        idx_sum = accumulated_sum

    logger.debug("Generated list of single excitations: %s", single_excitations)

    # we can find the actual list of excitations by doing the following:
    #   1. combine the single alpha- and beta-spin excitations
    #   2. find all possible combinations of length `num_excitations`
    pool = itertools.combinations(single_excitations, num_excitations)

    excitations = []
    visited_excitations = set()

    for exc in pool:
        # validate an excitation by asserting that all indices are unique:
        #   1. get the frozen set of indices in the excitation
        exc_set = frozenset(itertools.chain.from_iterable(exc))
        #   2. all indices must be unique (size of set equals 2 * num_excitations)
        #   3. and we also don't want to include permuted variants of identical excitations
        if len(exc_set) == num_excitations * 2 and exc_set not in visited_excitations:
            visited_excitations.add(exc_set)
            exc_tuple = tuple(zip(*exc))
            excitations.append(exc_tuple)
            logger.debug("Added the excitation: %s", exc_tuple)

    return excitations
