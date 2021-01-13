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

"""Qubit Mapping interface."""

from abc import ABC, abstractmethod

from qiskit.aqua.operators import PauliSumOp
from qiskit.chemistry.operators import ParticleOperator, SecondQuantizedOperator


class QubitMapping(ABC):
    """The interface for implementing methods which map from a `SecondQuantizedOperator` to a
    `PauliSumOp`.
    """

    @abstractmethod
    def supports_particle_type(self, particle_type: ParticleOperator) -> bool:
        """Returns whether the queried particle-type operator is supported by this mapping.

        Args:
            particle_type: the particle-type to query support for.

        Returns:
            A boolean indicating whether the queried particle-type is supported.
        """
        raise NotImplementedError()

    @abstractmethod
    def map(self, second_q_op: SecondQuantizedOperator) -> PauliSumOp:
        """Maps a `SecondQuantizedOperator` to a `PauliSumOp`.

        Args:
            second_q_op: the `SecondQuantizedOperator` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.
        """
        raise NotImplementedError()
