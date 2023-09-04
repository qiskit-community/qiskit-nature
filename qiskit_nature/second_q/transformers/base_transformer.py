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

"""The Base Operator Transformer interface."""

from abc import ABC, abstractmethod

from qiskit_nature.second_q.hamiltonians import Hamiltonian
from qiskit_nature.second_q.problems import BaseProblem


class BaseTransformer(ABC):
    """The interface for implementing methods which map from one
    :class:`~qiskit_nature.second_q.problems.BaseProblem` to another.
    These methods may affect the size of the Hilbert space.
    """

    @abstractmethod
    def transform(self, problem: BaseProblem) -> BaseProblem:
        """Transforms one :class:`~qiskit_nature.second_q.problems.BaseProblem` into another.
        This may affect the size of the Hilbert space.

        Args:
            problem: the problem to be transformed.

        Raises:
            NotImplementedError: when an unsupported problem type is provided.

        Returns:
            A new `BaseProblem` instance.
        """
        raise NotImplementedError()

    @abstractmethod
    def transform_hamiltonian(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        """Transforms one :class:`~qiskit_nature.second_q.hamiltonians.Hamiltonian` into another.
        This may affect the size of the Hilbert space.

        Args:
            hamiltonian: the hamiltonian to be transformed.

        Raises:
            NotImplementedError: when an unsupported hamiltonian type is provided.

        Returns:
            A new `Hamiltonian` instance.
        """
        raise NotImplementedError()
