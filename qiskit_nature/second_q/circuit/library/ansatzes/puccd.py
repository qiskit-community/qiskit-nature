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
The paired-UCCD Ansatz.
"""

from __future__ import annotations

import logging

from qiskit.circuit import QuantumCircuit
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.mappers import QubitMapper

from .ucc import UCC
from .utils.fermionic_excitation_generator import (
    generate_fermionic_excitations,
    get_alpha_excitations,
)

logger = logging.getLogger(__name__)


class PUCCD(UCC):
    """The PUCCD Ansatz.

    The PUCCD ansatz (by default) only contains double excitations. Furthermore, it enforces all
    excitations to occur in parallel in the alpha and beta species. For more information see also
    [1].

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
        include_imaginary: bool = False,
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
            include_imaginary: Boolean flag which when set to ``True`` expands the ansatz to include
                imaginary parts using twice the number of free parameters.

        Raises:
            QiskitNatureError: if the number of alpha and beta electrons is not equal.

        """
        self._validate_num_particles(num_particles)
        self._include_singles = include_singles
        super().__init__(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            excitations=self.generate_excitations,
            qubit_mapper=qubit_mapper,
            alpha_spin=True,
            beta_spin=True,
            max_spin_excitation=None,
            generalized=generalized,
            include_imaginary=include_imaginary,
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

    def generate_excitations(
        self, num_spatial_orbitals: int, num_particles: tuple[int, int]
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Generates the excitations for the PUCCD Ansatz.

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

        # generate alpha-spin orbital indices for occupied and unoccupied ones
        alpha_excitations = get_alpha_excitations(
            num_spatial_orbitals, num_electrons, generalized=self._generalized
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

    # TODO: when ooVQE gets refactored, it may turn out that this Ansatz can indeed by used for
    # unrestricted spin systems.
    def _validate_num_particles(self, num_particles):
        try:
            assert num_particles[0] == num_particles[1]
        except AssertionError as exc:
            raise QiskitNatureError(
                "The PUCCD Ansatz only works for singlet-spin systems. However, you specified "
                "differing numbers of alpha and beta electrons:",
                str(num_particles),
            ) from exc
