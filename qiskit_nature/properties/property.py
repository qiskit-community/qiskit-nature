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

from __future__ import annotations

from abc import ABC, abstractmethod, abstractclassmethod
from typing import List, Optional, Union

from qiskit_nature.drivers import QMolecule, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.results import EigenstateResult


class Property(ABC):
    """TODO."""

    def __init__(self, name: str, register_length: int) -> None:
        """TODO."""
        self._name = name
        self._register_length = register_length

    @property
    def name(self) -> str:
        """Returns the name."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Sets the name."""
        self._name = name

    @property
    def register_length(self) -> int:
        """Returns the register_length."""
        return self._register_length

    @register_length.setter
    def register_length(self, register_length: int) -> None:
        """Sets the register_length."""
        self._register_length = register_length

    @abstractclassmethod
    def from_driver_result(cls, result: Union[QMolecule, WatsonHamiltonian]) -> Property:
        """TODO."""

    @abstractmethod
    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """TODO."""

    @abstractmethod
    def interpret(self, result: EigenstateResult) -> None:
        """TODO."""
