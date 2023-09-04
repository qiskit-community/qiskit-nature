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

"""
The spin-adapted paired-UCC ansatz.
"""

from __future__ import annotations

import logging
from typing import Sequence, cast
from collections import defaultdict

from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.operators import FermionicOp

from .ucc import UCC
from .utils.fermionic_excitation_generator import (
    generate_fermionic_excitations,
    get_alpha_excitations,
)

logger = logging.getLogger(__name__)


class PUCCSD(UCC):
    """The spin-adapted paired-UCC Ansatz.

    This ansatz (by default) contains paired single and double excitations. This ensures that not
    only the number of particles but also the spin is preserved. [1]

    Note, that this ansatz will produce a generalized operator pool (``generalized=True``).

    This is a convenience subclass of the UCC ansatz. For more information refer to :class:`UCC`.

    References:
        [1] `arXiv:2207.00085 <https://arxiv.org/abs/2207.00085>`_

    """

    def __init__(
        self,
        num_spatial_orbitals: int | None = None,
        num_particles: tuple[int, int] | None = None,
        qubit_mapper: QubitMapper | None = None,
        *,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
    ) -> None:
        # pylint: disable=unused-argument
        """

        Args:
            num_spatial_orbitals: The number of spatial orbitals.
            num_particles: The tuple of the number of alpha- and beta-spin particles.
            qubit_mapper: The :class:`~qiskit_nature.second_q.mappers.QubitMapper` instance which
                takes care of mapping to a qubit operator.
            reps: The number of times to repeat the evolved operators.
            initial_state: A ``QuantumCircuit`` object to prepend to the circuit.
        """
        self._excitations_dict: dict[
            tuple[tuple[int, ...], tuple[int, ...]], list[tuple[tuple[int, ...], tuple[int, ...]]]
        ] | None = None
        super().__init__(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            excitations=self.generate_excitations,
            qubit_mapper=qubit_mapper,
            alpha_spin=True,
            beta_spin=True,
            max_spin_excitation=None,
            generalized=True,
            include_imaginary=False,
            reps=reps,
            initial_state=initial_state,
        )

    def generate_excitations(
        self, num_spatial_orbitals: int, num_particles: tuple[int, int]
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Generates the excitations for the PUCCSD Ansatz.

        Args:
            num_spatial_orbitals: the number of spatial orbitals.
            num_particles: the number of alpha and beta electrons. Note, these must be identical for
            this class.

        Returns:
            The list of excitations encoded as tuples of tuples. Each tuple in the list is a pair of
            tuples. The first tuple contains the occupied spin orbital indices whereas the second
            one contains the indices of the unoccupied spin orbitals.
        """
        excitations: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        excitations.extend(
            generate_fermionic_excitations(
                1,
                num_spatial_orbitals,
                num_particles,
                alpha_spin=True,
                beta_spin=False,
                generalized=True,
            )
        )

        num_electrons = num_particles[0]
        beta_index_shift = num_spatial_orbitals

        # generate alpha-spin orbital indices for occupied and unoccupied ones
        alpha_excitations = get_alpha_excitations(
            num_spatial_orbitals, num_electrons, generalized=True
        )
        logger.debug("Generated list of single alpha excitations: %s", alpha_excitations)

        for alpha_exc in alpha_excitations:
            # create the beta-spin excitation by shifting into the upper block-spin orbital indices
            beta_exc = (
                alpha_exc[0] + beta_index_shift,
                alpha_exc[1] + beta_index_shift,
            )
            # add the excitation tuple
            occ: tuple[int, ...]
            unocc: tuple[int, ...]
            occ, unocc = zip(alpha_exc, beta_exc)
            exc_tuple = (occ, unocc)
            excitations.append(exc_tuple)
            logger.debug("Added the excitation: %s", exc_tuple)

        return excitations

    def _build_fermionic_excitation_ops(self, excitations: Sequence) -> list[FermionicOp]:
        """Builds all possible excitation operators with the given number of excitations for the
        specified number of particles distributed in the number of orbitals.

        Args:
            excitations: the list of excitations.

        Returns:
            The list of excitation operators in the second quantized formalism.
        """
        operators: list[FermionicOp] = []
        self._excitations_dict = defaultdict(list)
        beta_index_shift = self.num_spatial_orbitals

        # Reform the excitations list to a dictionary. Each items in the dictionary
        # corresponds to a parameter.
        for exc in excitations:
            if len(exc[0]) == 1:
                # single excitation
                self._excitations_dict[exc].append(exc)
                self._excitations_dict[exc].append(
                    ((exc[0][0] + beta_index_shift,), (exc[1][0] + beta_index_shift,))
                )
            elif len(exc[0]) == 2:
                # double excitation
                self._excitations_dict[exc].append(exc)

        for exc_list in self._excitations_dict.values():
            sum_ops = cast(FermionicOp, sum(super()._build_fermionic_excitation_ops(exc_list)))
            operators.append(sum_ops)

        return operators
