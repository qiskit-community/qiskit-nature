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

"""Ground state computation using a minimum eigensolver."""

from __future__ import annotations

import logging

from qiskit_algorithms import MinimumEigensolver
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.problems import EigenstateResult

from .ground_state_solver import GroundStateSolver

LOGGER = logging.getLogger(__name__)


class GroundStateEigensolver(GroundStateSolver):
    """Ground state computation using a minimum eigensolver."""

    def __init__(
        self,
        qubit_mapper: QubitMapper,
        solver: MinimumEigensolver,
    ) -> None:
        # pylint: disable=unused-argument
        """
        Args:
            qubit_mapper: The :class:`~qiskit_nature.second_q.mappers.QubitMapper` instance that
                converts a second quantized operator to qubit operators.
            solver: Minimum Eigensolver object.
        """
        super().__init__(qubit_mapper)
        self._solver = solver

    @property
    def solver(self) -> MinimumEigensolver:
        return self._solver

    def supports_aux_operators(self):
        return self.solver.supports_aux_operators()

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> EigenstateResult:
        """Compute Ground State properties.

        Args:
            problem: A class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """
        main_operator, aux_ops = self.get_qubit_operators(problem, aux_operators)
        raw_mes_result = self.solver.compute_minimum_eigenvalue(main_operator, aux_ops)

        eigenstate_result = EigenstateResult.from_result(raw_mes_result)
        result = problem.interpret(eigenstate_result)
        return result

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> tuple[SparsePauliOp, dict[str, SparsePauliOp] | None]:
        # Note that ``aux_ops`` contains not only the transformed ``aux_operators`` passed by the
        # user but also additional ones from the transformation
        main_second_q_op, aux_second_q_ops = problem.second_q_ops()

        main_operator = self._qubit_mapper.map(main_second_q_op)
        aux_ops = self._qubit_mapper.map(aux_second_q_ops)

        if aux_operators is not None:
            for name_aux, aux_op in aux_operators.items():
                if isinstance(aux_op, SparseLabelOp):
                    converted_aux_op = self._qubit_mapper.map(aux_op)
                else:
                    converted_aux_op = aux_op

                if name_aux in aux_ops.keys():
                    LOGGER.warning(
                        "The key '%s' was already taken by an internally constructed auxiliary "
                        "operator! The internal operator was overridden by the one provided manually. "
                        "If this was not the intended behavior, please consider renaming "
                        "this operator.",
                        name_aux,
                    )
                if converted_aux_op is not None:
                    # The custom op overrides the default op if the key is already taken.
                    aux_ops[name_aux] = converted_aux_op
                else:
                    LOGGER.warning(
                        "The manually provided operator '%s' got reduced to `None` in the mapping "
                        "process. This can occur for example when it does not commute with the "
                        "hamiltonian after applying the determined symmetry reductions. Thus, this "
                        "operator will not be used!",
                        name_aux,
                    )

        # if the eigensolver does not support auxiliary operators, reset them
        if not self.solver.supports_aux_operators():
            aux_ops = None
        return main_operator, aux_ops
