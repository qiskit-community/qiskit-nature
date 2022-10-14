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

import qiskit_nature  # pylint: disable=unused-import
from qiskit_nature.second_q.operators import SecondQuantizedOp


class Hamiltonian(ABC):
    """The Hamiltonian base class is the main component of the :class:`.BaseProblem`.

    This class is a factory for a :class:`.SecondQuantizedOp`, accessible via its ``second_q_op()``
    method.
    """

    @abstractmethod
    def second_q_op(self) -> SecondQuantizedOp:
        """Generates the actual operator represented by this Hamiltonian.

        Returns:
            The :class:`.SecondQuantizedOp` form of this Hamiltonian.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def register_length(self) -> int | None:
        """The size of the operator generated by the :meth:`second_q_op` method."""
        raise NotImplementedError()

    def interpret(
        self, result: "qiskit_nature.second_q.problemsEigenstateResult"  # type: ignore[name-defined]
    ) -> None:
        """Interprets an :class:`~qiskit_nature.second_q.problems.EigenstateResult`
        in this hamiltonians context.

        Args:
            result: the result to add meaning to.
        """
