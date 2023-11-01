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

"""The ground state calculation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.problems import EigenstateResult


class GroundStateSolver(ABC):
    """The ground state calculation interface."""

    def __init__(
        self,
        qubit_mapper: QubitMapper,
    ) -> None:
        # pylint: disable=unused-argument
        """
        Args:
            qubit_mapper: The :class:`~qiskit_nature.second_q.mappers.QubitMapper` instance that
                converts a second quantized operator to qubit operators.
        """
        self._qubit_mapper = qubit_mapper

    @abstractmethod
    def solve(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> EigenstateResult:
        """Compute the ground state energy of the molecule that was supplied via the driver.

        Args:
            problem: A class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> tuple[SparsePauliOp, dict[str, SparsePauliOp] | None]:
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

    @abstractmethod
    def supports_aux_operators(self) -> bool:
        """Returns whether the eigensolver supports auxiliary operators."""
        raise NotImplementedError

    @property
    def qubit_mapper(self):
        """Returns the qubit mapper."""
        return self._qubit_mapper

    @property
    @abstractmethod
    def solver(self):
        """Returns the solver."""
