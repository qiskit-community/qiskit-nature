# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The calculation of excited states via an Eigensolver algorithm"""

import logging
from typing import List, Union, Optional, Any

from qiskit.algorithms import Eigensolver
from qiskit.opflow import PauliSumOp

from qiskit_nature import FermionicOperator
from qiskit_nature.results import (EigenstateResult,
                                   ElectronicStructureResult,
                                   VibronicStructureResult, )
from .excited_states_solver import ExcitedStatesSolver
from .eigensolver_factories import EigensolverFactory
from ...operators.second_quantization.qubit_converter import QubitConverter
from ...problems.second_quantization.base_problem import BaseProblem

logger = logging.getLogger(__name__)


class ExcitedStatesEigensolver(ExcitedStatesSolver):
    """The calculation of excited states via an Eigensolver algorithm"""

    def __init__(self, qubit_converter: QubitConverter,
                 solver: Union[Eigensolver, EigensolverFactory]) -> None:
        """

        Args:
            transformation: Qubit Operator Transformation
            solver: Minimum Eigensolver or MESFactory object.
        """
        self._qubit_converter = qubit_converter
        self._solver = solver

    @property
    def solver(self) -> Union[Eigensolver, EigensolverFactory]:
        """Returns the minimum eigensolver or factory."""
        return self._solver

    @solver.setter
    def solver(self, solver: Union[Eigensolver, EigensolverFactory]) -> None:
        """Sets the minimum eigensolver or factory."""
        self._solver = solver

    def solve(self, problem: BaseProblem,
              aux_operators: Optional[List[Any]] = None
              ) -> Union[ElectronicStructureResult, VibronicStructureResult]:
        """Compute Ground and Excited States properties.

        Args:
            aux_operators: Additional auxiliary operators to evaluate. Must be of type
                ``FermionicOperator`` if the qubit transformation is fermionic and of type
                ``BosonicOperator`` it is bosonic.

        Raises:
            NotImplementedError: If an operator in ``aux_operators`` is not of type
                ``FermionicOperator``.

        Returns:
            An eigenstate result. Depending on the transformation this can be an electronic
            structure or bosonic result.
        """
        if aux_operators is not None:
            if any(not isinstance(op, (PauliSumOp, FermionicOperator))
                   for op in aux_operators):
                raise NotImplementedError('Currently only fermionic problems are supported.')

        # get the operator and auxiliary operators, and transform the provided auxiliary operators
        # note that ``aux_operators`` contains not only the transformed ``aux_operators`` passed
        # by the user but also additional ones from the transformation
        second_q_ops = problem.second_q_ops()
        qubit_ops = self._qubit_converter.convert_match(second_q_ops)

        main_operator = qubit_ops[0]
        aux_operators = qubit_ops[1:]

        if isinstance(self._solver, EigensolverFactory):
            # this must be called after transformation.transform
            solver = self._solver.get_solver(problem)
        else:
            solver = self._solver

        # if the eigensolver does not support auxiliary operators, reset them
        if not solver.supports_aux_operators():
            aux_operators = None

        raw_es_result = solver.compute_eigenvalues(main_operator, aux_operators)

        eigenstate_result = EigenstateResult()
        eigenstate_result.raw_result = raw_es_result
        eigenstate_result.eigenenergies = raw_es_result.eigenvalues
        eigenstate_result.eigenstates = raw_es_result.eigenstates
        eigenstate_result.aux_operator_eigenvalues = raw_es_result.aux_operator_eigenvalues
        result = problem.interpret(eigenstate_result)
        return result
