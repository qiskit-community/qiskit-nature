# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The initial point interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_nature.second_q.problems import BaseProblem


class InitialPoint(ABC):
    r"""The initial point interface.

    The interface for utility classes that provide an initial point for the ``VQE`` parameters for a
    particular ``EvolvedOperatorAnsatz``.
    """

    @abstractmethod
    def __init__(self) -> None:
        self._ansatz: EvolvedOperatorAnsatz | None = None
        self._problem: BaseProblem | None = None

    @property
    @abstractmethod
    def ansatz(self) -> EvolvedOperatorAnsatz | None:
        """The evolved operator ansatz.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @ansatz.setter
    def ansatz(self, ansatz: EvolvedOperatorAnsatz) -> None:
        raise NotImplementedError

    @property
    def problem(self) -> BaseProblem | None:
        """The problem.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @problem.setter
    def problem(self, problem: BaseProblem) -> None:
        raise NotImplementedError

    @abstractmethod
    def to_numpy_array(self) -> np.ndarray:
        """Returns a NumPy array of the computed initial point.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def compute(
        self,
        ansatz: EvolvedOperatorAnsatz | None = None,
        problem: BaseProblem | None = None,
    ) -> None:
        """Compute the initial point array."""
        raise NotImplementedError
