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

    Interface for algorithms that can compute an initial point for the VQE parameters when using a
    UCC ansatz.
    """

    def __init__(self):
        self._grouped_property: GroupedSecondQuantizedProperty | None = None
        self._ansatz: UCC | None = None

    @property
    @abstractmethod
    def grouped_property(self) -> GroupedSecondQuantizedProperty:
        """The grouped property."""
        raise NotImplementedError

    @grouped_property.setter
    def grouped_property(self, grouped_property: GroupedSecondQuantizedProperty) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def ansatz(self) -> UCC:
        """The UCC ansatz."""
        raise NotImplementedError

    @ansatz.setter
    def ansatz(self, ansatz: UCC) -> None:
        raise NotImplementedError

    @abstractmethod
    def to_numpy_array(self) -> np.ndarray:
        """Returns a numpy array of the computed initial point."""
        raise NotImplementedError

    @abstractmethod
    def compute(
        self, grouped_property: GroupedSecondQuantizedProperty | None, ansatz: UCC | None
    ) -> np.ndarray:
        """Computes the initial point."""
        raise NotImplementedError
