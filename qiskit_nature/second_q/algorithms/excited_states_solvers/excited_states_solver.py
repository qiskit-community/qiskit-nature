# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
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
from typing import Optional, Union, Tuple

from qiskit.opflow import PauliSumOp

from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.problems import EigenstateResult


class ExcitedStatesSolver(ABC):
    """The excited states calculation interface."""

    @abstractmethod
    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[dict[str, Union[SparseLabelOp, PauliSumOp]]] = None,
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
        aux_operators: Optional[dict[str, Union[SparseLabelOp, PauliSumOp]]] = None,
    ) -> Tuple[PauliSumOp, Optional[dict[str, PauliSumOp]]]:
        """Construct qubit operators by getting the second quantized operators from the problem
        (potentially running a driver in doing so [can be computationally expensive])
        and using a QubitConverter to map and reduce the operators to qubit operators.

        Args:
            problem: A class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            Qubit operator.
            Additional auxiliary operators.
        """
