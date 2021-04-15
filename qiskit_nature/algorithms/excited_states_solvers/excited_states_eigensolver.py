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
from typing import List, Union, Optional

from qiskit.algorithms import Eigensolver
from qiskit.opflow import PauliSumOp

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.results import EigenstateResult

from .excited_states_solver import ExcitedStatesSolver
from .eigensolver_factories import EigensolverFactory

logger = logging.getLogger(__name__)


class ExcitedStatesEigensolver(ExcitedStatesSolver):
    """The calculation of excited states via an Eigensolver algorithm"""

    def __init__(self, qubit_converter: QubitConverter,
                 solver: Union[Eigensolver, EigensolverFactory]) -> None:
        """

        Args:
            qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The
                             Z2 symmetries stored in this instance are the basis for the
                             commutativity information returned by this method.
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
              aux_operators: Optional[List[Union[SecondQuantizedOp, PauliSumOp]]] = None,
              ) -> EigenstateResult:
        """Compute Ground and Excited States properties.

        Args:
            problem: a class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Raises:
            NotImplementedError: If an operator in ``aux_operators`` is not of type
                ``FermionicOperator``.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """
        # get the operator and auxiliary operators, and transform the provided auxiliary operators
        # note that ``aux_operators`` contains not only the transformed ``aux_operators`` passed
        # by the user but also additional ones from the transformation
        second_q_ops = problem.second_q_ops()

        main_operator = self._qubit_converter.convert(
            second_q_ops.main_operator,
            num_particles=problem.num_particles,
            sector_locator=problem.symmetry_sector_locator
        )
        aux_ops = self._qubit_converter.convert_match(second_q_ops.aux_operators_list)

        if aux_operators is not None:
            for aux_op in aux_operators:
                if isinstance(aux_op, SecondQuantizedOp):
                    aux_ops.append(self._qubit_converter.convert_match(aux_op, True))
                else:
                    aux_ops.append(aux_op)

        if isinstance(self._solver, EigensolverFactory):
            # this must be called after transformation.transform
            solver = self._solver.get_solver(problem)
        else:
            solver = self._solver

        # if the eigensolver does not support auxiliary operators, reset them
        if not solver.supports_aux_operators():
            aux_ops = None

        raw_es_result = solver.compute_eigenvalues(main_operator, aux_ops)

        eigenstate_result = EigenstateResult()
        eigenstate_result.raw_result = raw_es_result
        eigenstate_result.eigenenergies = raw_es_result.eigenvalues
        eigenstate_result.eigenstates = raw_es_result.eigenstates
        eigenstate_result.aux_operator_eigenvalues = raw_es_result.aux_operator_eigenvalues
        result = problem.interpret(eigenstate_result)
        return result
