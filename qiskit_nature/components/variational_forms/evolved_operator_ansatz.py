# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
TODO.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import logging

from qiskit.circuit.library import BlueprintCircuit
from qiskit.opflow import OperatorBase, EvolutionBase

logger = logging.getLogger(__name__)


class EvolvedOperatorAnsatz(BlueprintCircuit, ABC):
    """An implementation of the BlueprintCircuit to represent general evolved operators."""

    def __init__(self, ops: List[OperatorBase], reps: int,
                 evolution: Optional[EvolutionBase] = None):
        """

        Args:
            ops: the final list of operators to be evolved. All operators in this list should
            already have undergone symmetry reductions, etc.
            reps: the number of repitions of the circuit.
            evolution: the evolution to be used to evolve the operator.
        """
        self._ops = ops
        self._reps = reps
        # later, we should use Trotterization as the default evolution when this is None
        self._evolution = evolution

    @abstractmethod
    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        pass

    @abstractmethod
    def _build(self) -> None:
        pass
