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

import logging

from qiskit.circuit.library import BlueprintCircuit

logger = logging.getLogger(__name__)


class AdaptiveAnsatz(BlueprintCircuit, ABC):
    """An interface for adaptive Ansätze."""

    def push(self, index: int) -> None:
        """Pushes the building block with the given index onto the Circuit."""
        pass

    def pop(self) -> None:
        """Removes the last building block from the Circuit."""
        pass

    @abstractmethod
    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        pass

    @abstractmethod
    def _build(self) -> None:
        pass
