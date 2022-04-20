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

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from qiskit_nature.properties.second_quantization import GroupedSecondQuantizedProperty
from qiskit_nature.circuit.library import UCC


class InitialPoint(ABC):
    """The initial point interface.

    Interface for algorithms that can compute initial points for particular ansatzes.
    """

    def __init__(self):
        self._grouped_property: GroupedSecondQuantizedProperty = None
        self._ansatz: UCC = None

    @property
    @abstractmethod
    def initial_point(self) -> np.ndarray:
        """The initial point."""
        raise NotImplementedError

    @abstractmethod
    def get_initial_point(
        self, grouped_property: GroupedSecondQuantizedProperty | None, ansatz: UCC | None
    ) -> np.ndarray:
        """Computes the initial point."""
        raise NotImplementedError

    @property
    def grouped_property(self) -> GroupedSecondQuantizedProperty:
        """The grouped property."""
        return self._grouped_property

    @grouped_property.setter
    def grouped_property(self, grouped_property: GroupedSecondQuantizedProperty) -> None:
        """The grouped property."""
        self._grouped_property = grouped_property

    @property
    def ansatz(self) -> UCC:
        """The UCC ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UCC) -> None:
        """The UCC ansatz."""
        self._ansatz = ansatz
