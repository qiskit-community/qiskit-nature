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

"""TODO."""

from abc import ABC, abstractmethod
from typing import List


class VibrationalBasis(ABC):
    """TODO."""

    def __init__(
        self,
        num_modals_per_mode: List[int],
        threshold: float = 1e-6,
    ) -> None:
        """TODO."""
        self._num_modals_per_mode = num_modals_per_mode
        self._threshold = threshold

    @abstractmethod
    def _eval_integral(
        self,
        mode: int,
        modal_1: int,
        modal_2: int,
        power: int,
        kinetic_term: bool = False,
    ) -> float:
        """TODO."""
