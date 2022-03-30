# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The initial point interface."""

from abc import ABC, abstractmethod

import numpy as np

from qiskit_nature.properties.second_quantization import GroupedSecondQuantizedProperty
from qiskit_nature.circuit.library import UCC


class InitialPoint(ABC):
    """The initial point interface.

    Interface for algorithms that can compute initial points for particular ansatzes.
    """

    def __init__(self):
        self._driver_result: GroupedSecondQuantizedProperty = None
        self._ansatz: UCC = None

    @property
    @abstractmethod
    def initial_point(self) -> np.ndarray:
        """Returns the initial point."""
        raise NotImplementedError

    def get_initial_point(
        self, driver_result: GroupedSecondQuantizedProperty, ansatz: UCC
    ) -> np.ndarray:
        """Computes the initial point."""
        raise NotImplementedError
