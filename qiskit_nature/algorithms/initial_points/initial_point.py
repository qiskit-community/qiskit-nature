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

from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_nature.properties.second_quantization import GroupedSecondQuantizedProperty


class InitialPoint(ABC):
    r"""The initial point interface.

    The interface for utility classes that can compute an initial point for the ``VQE`` parameters
    for a particular ``EvolvedOperatorAnsatz``.
    """

    @abstractmethod
    def __init__(self):
        self._ansatz: EvolvedOperatorAnsatz | None = None
        self._grouped_property: GroupedSecondQuantizedProperty | None = None

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the computed initial point array.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

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
    @abstractmethod
    def grouped_property(self) -> GroupedSecondQuantizedProperty | None:
        """The grouped property.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @grouped_property.setter
    def grouped_property(self, grouped_property: GroupedSecondQuantizedProperty) -> None:
        raise NotImplementedError

    @abstractmethod
    def to_numpy_array(self) -> np.ndarray:
        """Returns a numpy array of the computed initial point.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        ansatz: EvolvedOperatorAnsatz | None,
        grouped_property: GroupedSecondQuantizedProperty | None,
    ) -> np.ndarray:
        """Computes the initial point.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def get_energy_corrections(self) -> np.ndarray:
        """The energy correction corresponding to each parameter value.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def get_energy_correction(self) -> float:
        """Returns the overall energy correction (zero)."""
        return 0.0

    def get_energy(self) -> float:
        """Returns the absolute energy (zero)."""
        return 0.0
