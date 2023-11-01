# This code is part of a Qiskit project.
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
The SUCCD Ansatz.
"""

from __future__ import annotations

from typing import Sequence, cast
from collections import defaultdict

import itertools
import logging

from qiskit.circuit import QuantumCircuit
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.mappers import QubitMapper

from qiskit_nature.second_q.operators import FermionicOp

from .ucc import UCC
from .utils.fermionic_excitation_generator import (
    generate_fermionic_excitations,
    get_alpha_excitations,
)

logger = logging.getLogger(__name__)


class SUCCD(UCC):
    """The SUCCD Ansatz.

    The SUCCD ansatz (by default) only contains double excitations. Furthermore, it only considers
    the set of excitations which is symmetrically invariant with respect to spin-flips of both
    particles. For more information see also [1].

    Note, that this ansatz can only work for singlet-spin systems. Therefore, the number of alpha
    and beta electrons must be equal.

    This is a convenience subclass of the UCC ansatz. For more information refer to :class:`UCC`.

    References:
        [1] `arXiv:1911.10864 <https://arxiv.org/abs/1911.10864>`_

    """

    def __init__(
        self,
        num_spatial_orbitals: int | None = None,
        num_particles: tuple[int, int] | None = None,
        qubit_mapper: QubitMapper | None = None,
        *,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        include_singles: tuple[bool, bool] = (False, False),
        generalized: bool = False,
        mirror: bool = False,
    ) -> None:
        # pylint: disable=unused-argument
        """
        Args:
            num_spatial_orbitals: The number of spatial orbitals.
            num_particles: The tuple of the number of alpha- and beta-spin particles.
            qubit_mapper: The :class:`~qiskit_nature.second_q.mappers.QubitMapper` which takes care
                of mapping to a qubit operator.
            reps: The number of times to repeat the evolved operators.
            initial_state: A ``QuantumCircuit`` object to prepend to the circuit.
            include_singles: enables the inclusion of single excitations per spin species.
            generalized: Boolean flag whether or not to use generalized excitations, which ignore
                the occupation of the spin orbitals. As such, the set of generalized excitations is
                only determined from the number of spin orbitals and independent from the number of
                particles.
            mirror: Boolean flag whether or not to include the symmetrically mirrored double
                excitations, while keeping the original number of circuit
                parameters. This results in mirrored excitations having identical parameter values.
                Enabling this parameter will result in the SUCCD ansatz referred to as
                "q-UCCSD0-full" in reference [1].

        Raises:
            QiskitNatureError: if the number of alpha and beta electrons is not equal.
        """
        self._validate_num_particles(num_particles)
        self._include_singles = include_singles
        self._mirror = mirror
        self._excitations_dict: dict[
            str, list[tuple[tuple[int, ...], tuple[int, ...]]]
        ] | None = None
        super().__init__(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            excitations=self.generate_excitations,
            qubit_mapper=qubit_mapper,
            alpha_spin=True,
            beta_spin=True,
            max_spin_excitation=None,
            generalized=generalized,
            reps=reps,
            initial_state=initial_state,
        )

    @property
    def include_singles(self) -> tuple[bool, bool]:
        """Whether to include single excitations."""
        return self._include_singles

    @include_singles.setter
    def include_singles(self, include_singles: tuple[bool, bool]) -> None:
        """Sets whether to include single excitations."""
        self._operators = None
        self._invalidate()
        self._include_singles = include_singles

    @property
    def mirror(self) -> bool:
        """Whether to include the symmetrically mirrored double excitations."""
        return self._mirror

    @mirror.setter
    def mirror(self, mirror: bool) -> None:
        """Sets whether to include the symmetrically mirrored double excitations."""
        self._operators = None
        self._invalidate()
        self._mirror = mirror

    def _filter_operators(self, operators):
        valid_operators, valid_excitations = [], []
        for op, ex in zip(operators, self._excitations_dict.values()):
            if op is not None:
                valid_operators.append(op)
                valid_excitations.extend(ex)

        self._excitation_list = valid_excitations
        self.operators = valid_operators

    def generate_excitations(
        self, num_spatial_orbitals: int, num_particles: tuple[int, int]
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Generates the excitations for the SUCCD Ansatz.

        Args:
            num_spatial_orbitals: the number of spatial orbitals.
            num_particles: the number of alpha and beta electrons. Note, these must be identical for
                this class.

        Raises:
            QiskitNatureError: if the number of alpha and beta electrons is not equal.

        Returns:
            The list of excitations encoded as tuples of tuples. Each tuple in the list is a pair of
            tuples. The first tuple contains the occupied spin orbital indices whereas the second
            one contains the indices of the unoccupied spin orbitals.
        """
        self._validate_num_particles(num_particles)

        excitations: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        excitations.extend(
            generate_fermionic_excitations(
                1,
                num_spatial_orbitals,
                num_particles,
                alpha_spin=self.include_singles[0],
                beta_spin=self.include_singles[1],
            )
        )
        num_electrons = num_particles[0]
        beta_index_shift = num_spatial_orbitals

        if self._mirror:
            # We can use `generate_fermionic_excitations` here because we want to include the
            # symmetrically mirrored double excitations
            excitations.extend(
                generate_fermionic_excitations(
                    2,
                    num_spatial_orbitals,
                    num_particles,
                    max_spin_excitation=1,
                    generalized=self._generalized,
                )
            )

        else:
            # generate alpha-spin orbital indices for occupied and unoccupied ones
            alpha_excitations = get_alpha_excitations(
                num_spatial_orbitals, num_electrons, generalized=self._generalized
            )
            logger.debug("Generated list of single alpha excitations: %s", alpha_excitations)

            # Find all possible double excitations constructed from the list of single excitations.
            # Note, that we use `combinations_with_replacement` here, in order to also get those
            # double excitations which excite from the same occupied level twice. We will need
            # those in the following post-processing step.

            for exc in itertools.combinations_with_replacement(alpha_excitations, 2):
                # find the two excitations (Note: SUCCD only works for double excitations!)
                alpha_exc, second_exc = exc[0], exc[1]
                # shift the second excitation into the beta-spin orbital index range
                beta_exc = (
                    second_exc[0] + beta_index_shift,
                    second_exc[1] + beta_index_shift,
                )
                # add the excitation tuple
                occ: tuple[int, ...]
                unocc: tuple[int, ...]
                occ, unocc = zip(alpha_exc, beta_exc)
                exc_tuple = (occ, unocc)
                excitations.append(exc_tuple)
                logger.debug("Added the excitation: %s", exc_tuple)

        return excitations

    def _validate_num_particles(self, num_particles):
        try:
            assert num_particles[0] == num_particles[1]
        except AssertionError as exc:
            raise QiskitNatureError(
                "The SUCCD Ansatz only works for singlet-spin systems. However, you specified "
                "differing numbers of alpha and beta electrons:",
                str(num_particles),
            ) from exc

    def _build_fermionic_excitation_ops(self, excitations: Sequence) -> list[FermionicOp]:
        """Builds all possible excitation operators with the given number of excitations for the
        specified number of particles distributed in the number of orbitals.

        Args:
            excitations: the list of excitations.

        Returns:
            The list of excitation operators in the second quantized formalism.
        """
        operators: list[FermionicOp] = []
        excitations_dictionary: dict[
            str, list[tuple[tuple[int, ...], tuple[int, ...]]]
        ] = defaultdict(list)
        beta_index_shift = self.num_spatial_orbitals

        # Reform the excitations list to a dictionary. Each items in the dictionary
        # corresponds to a parameter.
        for exc in excitations:
            alpha_occ = exc[0][0]
            # beta occupied indices. If include singles, then beta_occ=alpha_occ
            beta_occ = exc[0][-1]
            alpha_unocc = exc[1][0]
            # beta unoccupied indices. If include singles, then beta_unocc=alpha_unocc
            beta_unocc = exc[1][-1]
            alpha_exc = int(
                str(alpha_occ) + str(alpha_unocc)
            )  # alpha occupied and unoccupied indices
            beta_exc = int(
                str(abs(beta_occ - beta_index_shift)) + str(abs(beta_unocc - beta_index_shift))
            )  # beta occupied and unoccupied indices.

            exc_level = str(abs(alpha_exc - beta_exc)) + str(alpha_exc + beta_exc)
            # exc_level is a 4-number string, which indicate alpha and beta occupied+unoccupied indices.
            # Thus, the level of an excitations is indicated by this string. The symmetrically
            # mirrored double excitations have the same exc_level string, and the
            # excitations with the same level will be assigned the same parameter.

            excitations_dictionary[exc_level].append(exc)

        self._excitations_dict = excitations_dictionary

        for exc_level, exc_level_items in excitations_dictionary.items():
            sum_ops = cast(
                FermionicOp, sum(super()._build_fermionic_excitation_ops(exc_level_items))
            )
            operators.append(sum_ops)

        return operators
