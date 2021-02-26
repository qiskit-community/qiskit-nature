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

from typing import Tuple

import itertools
import logging

logger = logging.getLogger(__name__)


class ExcitationOpBuilder:
    """A factory class to construct exctiation operators."""

    @staticmethod
    def build_excitation_ops(num_excitations: int,
                             num_orbitals: int,
                             num_particles: Tuple[int, int]
                             ) -> None:
        """Builds all possible excitation operators with the given number of excitations for the
        specified number of particles distributed in the number of orbitals.

        This method assumes block-ordered spin-orbitals.

        Args:
            num_excitations: number of excitations per operator (1 means single excitations, etc.).
            num_orbitals: number of spin-orbitals.
            num_particles: number of alpha and beta particles.
        """
        # generate sets of alpha-spin orbital indices for occupied and unoccupied ones
        set_alpha_occ = set(range(num_particles[0]))
        set_alpha_unocc = set(range(num_particles[0], num_orbitals // 2))
        # the cartesian product of these sets gives all possible single alpha-spin excitations
        alpha_excitations = set(itertools.product(set_alpha_occ, set_alpha_unocc))

        # generate sets of beta-spin orbital indices for occupied and unoccupied ones
        set_beta_occ = set(range(num_orbitals // 2, num_particles[1] + num_orbitals // 2))
        set_beta_unocc = set(range(num_orbitals // 2 + num_particles[1], num_orbitals))
        # the cartesian product of these sets gives all possible single beta-spin excitations
        beta_excitations = set(itertools.product(set_beta_occ, set_beta_unocc))

        excitations = set()
        # we can find the actual list of excitations by doing the following:
        #   1. combine the sets of single alpha- and beta-spin excitations
        #   2. find all possible combinations of length `num_excitations`
        #   3. only use those where all indices are unique
        for comb in itertools.combinations(alpha_excitations | beta_excitations, num_excitations):
            # validate a combination by asserting that all indices are unique:
            #   1. flatten the tuple of tuples (chain.from_iterable)
            #   2. the length equals twice the number of excitations (one excitation gives to
            #      indices)
            if len(set(itertools.chain.from_iterable(comb))) == num_excitations * 2:
                # we zip the tuple of tuples to obtain a new one which has the structure:
                #   ((occupied indices), (unoccupied indices))
                excitations.add(tuple(zip(*comb)))

        # Some notes in terms of customization:
        #   - which excitations are present can (obviously) be configured via `num_excitations`
        #   - we can add an option to limit it to only `alpha` or only `beta` easily
        #   - if we want both spin kinds but no cross terms (i.e. pure spin excitations), we need to
        #     construct the excitation list in two separate steps. First only alpha, then only beta.
        #     But this is also easy to adjust.

        for exc in excitations:
            label = ['I'] * num_orbitals
            for occ in exc[0]:
                label[occ] = '-'
            for unocc in exc[1]:
                label[unocc] = '+'
            label = ''.join(label)
            print(exc, label)
