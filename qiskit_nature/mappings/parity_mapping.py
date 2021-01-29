# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Parity Mapping interface."""

from qiskit.aqua.operators import PauliSumOp

from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.operators.second_quantization.particle_op import ParticleOp

from .qubit_mapping import QubitMapping


class ParityMapping(QubitMapping):
    """The Parity fermion-to-qubit mapping. """

    def supports_particle_type(self, particle_type: ParticleOp) -> bool:
        """Returns whether the queried particle-type operator is supported by this mapping.

        Args:
            particle_type: the particle-type to query support for.

        Returns:
            A boolean indicating whether the queried particle-type is supported.
        """
        return isinstance(particle_type, FermionicOp)

    def map(self, second_q_op: ParticleOp) -> PauliSumOp:
        """Maps a `ParticleOp` to a `PauliSumOp` using the Parity fermion-to-qubit
        mapping.

        Args:
            second_q_op: the `ParticleOp` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.
        """
        # TODO
        raise NotImplementedError()
