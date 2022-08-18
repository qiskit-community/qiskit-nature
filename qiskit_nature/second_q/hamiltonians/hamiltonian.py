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

"""The Hamiltonian base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from qiskit_nature.second_q.operators import SecondQuantizedOp

if TYPE_CHECKING:
    from qiskit_nature.second_q.problems import EigenstateResult


class Hamiltonian(ABC):
    """TODO."""

    @abstractmethod
    def second_q_op(self) -> SecondQuantizedOp:
        """TODO."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def register_length(self) -> int:
        """TODO."""
        raise NotImplementedError()

    def interpret(self, result: "EigenstateResult") -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in this hamiltonians context.

        Args:
            result: the result to add meaning to.
        """
