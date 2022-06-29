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
"""An interface for sampling problems."""
from abc import ABC, abstractmethod
from typing import Union

from qiskit.opflow import PauliSumOp, PauliOp
from qiskit.algorithms import MinimumEigensolverResult

from qiskit_nature.second_q.problems import EigenstateResult


class SamplingProblem(ABC):
    """An interface for sampling problems."""

    @abstractmethod
    def qubit_op(self) -> Union[PauliOp, PauliSumOp]:
        """Returns a qubit operator that represents a Hamiltonian encoding the sampling problem."""
        pass

    @abstractmethod
    def interpret(self, raw_result: MinimumEigensolverResult) -> EigenstateResult:
        """Interprets results of an optimization."""
        pass
