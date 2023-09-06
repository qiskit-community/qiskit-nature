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
The UCCSD Ansatz.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.mappers import QubitMapper
from .ucc import UCC


class UCCSD(UCC):
    """The UCCSD Ansatz.

    This is a convenience subclass of the UCC ansatz. For more information refer to :class:`UCC`.
    """

    def __init__(
        self,
        num_spatial_orbitals: int | None = None,
        num_particles: tuple[int, int] | None = None,
        qubit_mapper: QubitMapper | None = None,
        *,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        generalized: bool = False,
        preserve_spin: bool = True,
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
            generalized: Boolean flag whether or not to use generalized excitations, which ignore
                the occupation of the spin orbitals. As such, the set of generalized excitations is
                only determined from the number of spin orbitals and independent from the number of
                particles.
            preserve_spin: Boolean flag whether or not to preserve the particle spins.
            include_imaginary: Boolean flag which when set to ``True`` expands the ansatz to include
                imaginary parts using twice the number of free parameters.
        """
        super().__init__(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            excitations="sd",
            qubit_mapper=qubit_mapper,
            alpha_spin=True,
            beta_spin=True,
            max_spin_excitation=None,
            generalized=generalized,
            preserve_spin=preserve_spin,
            include_imaginary=include_imaginary,
            reps=reps,
            initial_state=initial_state,
        )
