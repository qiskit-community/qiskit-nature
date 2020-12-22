# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Direct Mapping interface."""

from qiskit.aqua.operators import PauliSumOp
from qiskit.chemistry.operators import BosonicOperator, ParticleOperator, SecondQuantizedOperator

from .qubit_mapping import QubitMapping


class DirectMapping(QubitMapping):
    """The Direct boson-to-qubit mapping. """

    def supports_particle_type(self, particle_type: ParticleOperator) -> bool:
        """Returns whether the queried particle-type operator is supported by this mapping.

        Args:
            particle_type: the particle-type to query support for.

        Returns:
            A boolean indicating whether the queried particle-type is supported.
        """
        if isinstance(particle_type, BosonicOperator):
            return True
        return False

    def map(self, second_q_op: SecondQuantizedOperator) -> PauliSumOp:
        """Maps a `SecondQuantizedOperator` to a `PauliSumOp` using the Direct boson-to-qubit
        mapping.

        Args:
            second_q_op: the `SecondQuantizedOperator` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.
        """
        # TODO
        raise NotImplementedError()
