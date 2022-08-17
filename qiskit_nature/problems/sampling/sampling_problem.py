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

from qiskit_nature.results import EigenstateResult
from ...deprecation import warn_deprecated, DeprecatedType, NatureDeprecationWarning


class SamplingProblem(ABC):
    """An interface for sampling problems."""

    def __init__(self):
        warn_deprecated(
            "0.5.0",
            old_type=DeprecatedType.CLASS,
            old_name="qiskit_nature.problems.sampling.SamplingProblem",
            additional_msg="This class is being removed from Qiskit Nature",
            stack_level=3,
            category=NatureDeprecationWarning,
        )

    @abstractmethod
    def qubit_op(self) -> Union[PauliOp, PauliSumOp]:
        """Returns a qubit operator that represents a Hamiltonian encoding the sampling problem."""
        pass

    @abstractmethod
    def interpret(self, raw_result: MinimumEigensolverResult) -> EigenstateResult:
        """Interprets results of an optimization."""
        pass
