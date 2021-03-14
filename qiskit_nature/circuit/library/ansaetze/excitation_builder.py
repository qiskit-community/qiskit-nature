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

from typing import List, Tuple

import itertools
import logging

from qiskit_nature.operators.second_quantization import FermionicOp, SecondQuantizedOp

logger = logging.getLogger(__name__)


class ExcitationBuilder:
    """A factory class to construct excitation operators."""

    @staticmethod
    def build_excitation_ops(num_excitations: int,
                             num_spin_orbitals: int,
                             num_particles: Tuple[int, int]
                             ) -> List[SecondQuantizedOp]:
        """Builds all possible excitation operators with the given number of excitations for the
        specified number of particles distributed in the number of orbitals.

        This method assumes block-ordered spin-orbitals.

        Args:
            num_excitations: number of excitations per operator (1 means single excitations, etc.).
            num_spin_orbitals: number of spin-orbitals.
            num_particles: number of alpha and beta particles.

        Returns:
            The list of excitation operators in the second quantized formalism.
        """
        # generate alpha-spin orbital indices for occupied and unoccupied ones
        alpha_occ = list(range(num_particles[0]))
        alpha_unocc = list(range(num_particles[0], num_spin_orbitals // 2))
        # the Cartesian product of these lists gives all possible single alpha-spin excitations
        alpha_excitations = list(itertools.product(alpha_occ, alpha_unocc))

        # generate beta-spin orbital indices for occupied and unoccupied ones
        beta_occ = list(range(num_spin_orbitals // 2, num_particles[1] + num_spin_orbitals // 2))
        beta_unocc = list(range(num_spin_orbitals // 2 + num_particles[1], num_spin_orbitals))
        # the Cartesian product of these lists gives all possible single beta-spin excitations
        beta_excitations = list(itertools.product(beta_occ, beta_unocc))

        excitations = list()
        # we can find the actual list of excitations by doing the following:
        #   1. combine the single alpha- and beta-spin excitations
        #   2. find all possible combinations of length `num_excitations`
        #   3. only use those where all indices are unique
        for comb in itertools.combinations(alpha_excitations + beta_excitations, num_excitations):
            # validate a combination by asserting that all indices are unique:
            #   1. flatten the tuple of tuples (chain.from_iterable)
            #   2. the length equals twice the number of excitations (one excitation gives to
            #      indices)
            if len(set(itertools.chain.from_iterable(comb))) == num_excitations * 2:
                # we zip the tuple of tuples to obtain a new one which has the structure:
                #   ((occupied indices), (unoccupied indices))
                excitations.append(tuple(zip(*comb)))

        # Some notes in terms of customization:
        #   - which excitations are present can (obviously) be configured via `num_excitations`
        #   - we can add an option to limit it to only `alpha` or only `beta` easily
        #   - if we want both spin kinds but no cross terms (i.e. pure spin excitations), we need to
        #     construct the excitation list in two separate steps. First only alpha, then only beta.
        #     But this is also easy to adjust.

        operators = []
        for exc in excitations:
            label = ['I'] * num_spin_orbitals
            for occ in exc[0]:
                label[occ] = '+'
            for unocc in exc[1]:
                label[unocc] = '-'
            op = FermionicOp(''.join(label))
            op -= op.adjoint()
            # we need to account for an additional imaginary phase in the exponent (see also
            # `PauliTrotterEvolution.convert`)
            op *= 1j
            operators.append(SecondQuantizedOp([op]))

        return operators
