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

"""The minimum eigensolver factory for ground state calculation algorithms."""

from abc import ABC, abstractmethod

from qiskit.algorithms import MinimumEigensolver

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import BaseProblem


class MinimumEigensolverFactory(ABC):
    """A factory to construct a minimum eigensolver based on a qubit operator transformation."""

    @abstractmethod
    def get_solver(
        self, problem: BaseProblem, qubit_converter: QubitConverter
    ) -> MinimumEigensolver:
        """Returns a minimum eigensolver, based on the qubit operator transformation.

        Args:
            problem: a class encoding a problem to be solved.
            qubit_converter: a class that converts second quantized operator to qubit operator
                             according to a mapper it is initialized with.

        Returns:
            A minimum eigensolver suitable to compute the ground state of the molecule.
        """
        raise NotImplementedError

    @abstractmethod
    def supports_aux_operators(self) -> bool:
        """Returns whether the eigensolver generated by this factory supports auxiliary operators."""
        raise NotImplementedError
