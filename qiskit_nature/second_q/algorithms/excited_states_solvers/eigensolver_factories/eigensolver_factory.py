# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The eigensolver factory for excited states calculation algorithms."""

from abc import ABC, abstractmethod

from qiskit.algorithms import Eigensolver

from qiskit_nature.second_q.problems.base_problem import BaseProblem


class EigensolverFactory(ABC):
    """A factory to construct a eigensolver based on a qubit operator transformation."""

    @abstractmethod
    def get_solver(self, problem: BaseProblem) -> Eigensolver:
        """Returns a eigensolver, based on the qubit operator transformation.

        Args:
            problem: a class encoding a problem to be solved.

        Returns:
            An eigensolver suitable to compute the excited states of the molecule transformed
            by ``transformation``.
        """
        raise NotImplementedError
