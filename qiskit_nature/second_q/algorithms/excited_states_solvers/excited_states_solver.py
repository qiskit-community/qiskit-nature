# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The excited states calculation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.problems import EigenstateResult


class ExcitedStatesSolver(ABC):
    """The excited states calculation interface."""

    @abstractmethod
    def solve(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> EigenstateResult:
        r"""Compute the excited states energies of the molecule that was supplied via the driver.

        Args:
            problem: A class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def solver(self):
        """Returns the solver."""

    @abstractmethod
    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> tuple[SparseLabelOp, dict[str, SparseLabelOp] | None]:
        """Gets the operator and auxiliary operators, and transforms the provided auxiliary operators
        using a ``QubitMapper``.
        If the user-provided ``aux_operators`` contain a name which clashes with an internally
        constructed auxiliary operator, then the corresponding internal operator will be overridden by
        the user-provided operator.

        Args:
            problem:  A class encoding a problem defining the qubit operators.
            aux_operators: Additional auxiliary operators to transform.

        Returns:
            A tuple with the main operator (hamiltonian) and a dictionary of auxiliary default and
            custom operators.
        """
