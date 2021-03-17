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
TODO.
"""

from typing import List, Tuple, Optional

import itertools
import logging

from qiskit_nature.operators.second_quantization import FermionicOp, SecondQuantizedOp

logger = logging.getLogger(__name__)


def build_excitation_ops(num_excitations: int,
                         num_spin_orbitals: int,
                         num_particles: Tuple[int, int],
                         alpha_spin: bool = True,
                         beta_spin: bool = True,
                         max_spin_excitation: Optional[int] = None,
                         ) -> List[SecondQuantizedOp]:
    """Builds all possible excitation operators with the given number of excitations for the
    specified number of particles distributed in the number of orbitals.

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
        The list of excitation operators in the second quantized formalism.
    """
    alpha_excitations = []
    if alpha_spin:
        # generate alpha-spin orbital indices for occupied and unoccupied ones
        alpha_occ = list(range(num_particles[0]))
        alpha_unocc = list(range(num_particles[0], num_spin_orbitals // 2))
        # the Cartesian product of these lists gives all possible single alpha-spin excitations
        alpha_excitations = list(itertools.product(alpha_occ, alpha_unocc))

    beta_excitations = []
    if beta_spin:
        # generate beta-spin orbital indices for occupied and unoccupied ones
        beta_occ = list(range(num_spin_orbitals // 2,
                              num_spin_orbitals // 2 + num_particles[1]))
        beta_unocc = list(range(num_spin_orbitals // 2 + num_particles[1], num_spin_orbitals))
        # the Cartesian product of these lists gives all possible single beta-spin excitations
        beta_excitations = list(itertools.product(beta_occ, beta_unocc))

    excitations = list()
    visited_excitations = set()

    def add_excitations(pool):
        for exc in pool:
            # validate an excitation by asserting that all indices are unique:
            #   1. get the frozen set of indices in the excitation
            exc_set = frozenset(itertools.chain.from_iterable(exc))
            #   2. all indicies must be unique (size of set equals 2 * num_excitations)
            #   3. and we also don't want to include permuted variants of identical excitations
            if len(exc_set) == num_excitations * 2 and exc_set not in visited_excitations:
                visited_excitations.add(exc_set)
                excitations.append(tuple(zip(*exc)))

    # we can find the actual list of excitations by doing the following:
    #   1. combine the single alpha- and beta-spin excitations
    #   2. find all possible combinations of length `num_excitations`
    pool = itertools.combinations(alpha_excitations + beta_excitations, num_excitations)

    # if max_spin_excitation is set, we need to filter the pool of excitations
    if max_spin_excitation is not None:
        # first, remove all those excitations, in which more than max_spin_excitation alpha
        # excitations are performed at ones
        if alpha_excitations:  # False if empty list
            alpha_exc_set = set(alpha_excitations)
            pool = itertools.filterfalse(
                lambda exc: len(set(exc) & alpha_exc_set) > max_spin_excitation,
                pool
            )
        # then, do the same for beta
        if beta_excitations:  # False if empty list
            beta_exc_set = set(beta_excitations)
            pool = itertools.filterfalse(
                lambda exc: len(set(exc) & beta_exc_set) > max_spin_excitation,
                pool
            )

    add_excitations(pool)

    operators = []
    for exc in excitations:
        label = ['I'] * num_spin_orbitals
        for occ in exc[0]:
            label[occ] = '+'
        for unocc in exc[1]:
            label[unocc] = '-'
        op = FermionicOp(''.join(label))
        # TODO: this is UCC specific. Do we want to keep this here and make this a UCC-specific
        # excitation builder?
        op -= op.adjoint()
        # we need to account for an additional imaginary phase in the exponent (see also
        # `PauliTrotterEvolution.convert`)
        op *= 1j
        operators.append(op)

    return operators
