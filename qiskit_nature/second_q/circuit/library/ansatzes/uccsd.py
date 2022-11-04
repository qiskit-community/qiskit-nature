# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
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
from qiskit_nature.second_q.mappers import QubitConverter
from .ucc import UCC


class UCCSD(UCC):
    """The UCCSD Ansatz.

    This is a convenience subclass of the UCC ansatz. For more information refer to :class:`UCC`.
    """

    def __init__(
        self,
        num_spatial_orbitals: int | None = None,
        num_particles: tuple[int, int] | None = None,
        qubit_converter: QubitConverter | None = None,
        *,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        generalized: bool = False,
        preserve_spin: bool = True,
    ):
        """
        Args:
            num_spatial_orbitals: The number of spatial orbitals.
            num_particles: The tuple of the number of alpha- and beta-spin particles.
            qubit_converter: The :class:`~qiskit_nature.second_q.mappers.QubitConverter` instance
                which takes care of mapping to a qubit operator.
            reps: The number of times to repeat the evolved operators.
            initial_state: A ``QuantumCircuit`` object to prepend to the circuit.
            generalized: Boolean flag whether or not to use generalized excitations, which ignore
                the occupation of the spin orbitals. As such, the set of generalized excitations is
                only determined from the number of spin orbitals and independent from the number of
                particles.
            preserve_spin: Boolean flag whether or not to preserve the particle spins.
        """
        super().__init__(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            excitations="sd",
            qubit_converter=qubit_converter,
            alpha_spin=True,
            beta_spin=True,
            max_spin_excitation=None,
            generalized=generalized,
            preserve_spin=preserve_spin,
            reps=reps,
            initial_state=initial_state,
        )
