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

"""Ground state computation using a minimum eigensolver."""

from typing import Union, List, Optional, Dict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit.algorithms import MinimumEigensolver
from qiskit.opflow import OperatorBase, PauliSumOp, StateFn, CircuitSampler

from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.results import EigenstateResult
from .ground_state_solver import GroundStateSolver
from .minimum_eigensolver_factories import MinimumEigensolverFactory


class GroundStateEigensolver(GroundStateSolver):
    """Ground state computation using a minimum eigensolver."""

    def __init__(
        self,
        qubit_converter: QubitConverter,
        solver: Union[MinimumEigensolver, MinimumEigensolverFactory],
    ) -> None:
        """

        Args:
            qubit_converter: a class that converts second quantized operator to qubit operator
                             according to a mapper it is initialized with.
            solver: Minimum Eigensolver or MESFactory object, e.g. the VQEUCCSDFactory.
        """
        super().__init__(qubit_converter)
        self._solver = solver

    @property
    def solver(self) -> Union[MinimumEigensolver, MinimumEigensolverFactory]:
        """Returns the minimum eigensolver or factory."""
        return self._solver

    @solver.setter
    def solver(self, solver: Union[MinimumEigensolver, MinimumEigensolverFactory]) -> None:
        """Sets the minimum eigensolver or factory."""
        self._solver = solver

    def returns_groundstate(self) -> bool:
        """Whether the eigensolver returns the ground state or only ground state energy."""
        return self._solver.supports_aux_operators()

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[List[Union[SecondQuantizedOp, PauliSumOp]]] = None,
    ) -> EigenstateResult:
        """Compute Ground State properties.

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
        # note that ``aux_ops`` contains not only the transformed ``aux_operators`` passed by the
        # user but also additional ones from the transformation
        second_q_ops = problem.second_q_ops()

        if isinstance(second_q_ops, list):
            main_second_q_op = second_q_ops[0]
            aux_second_q_ops = second_q_ops[1:]
        elif isinstance(second_q_ops, dict):
            main_second_q_op = second_q_ops.pop(problem.main_property_name, None)
            if main_second_q_op is None:
                raise ValueError("TODO")
            aux_second_q_ops = second_q_ops
        else:
            raise TypeError("TODO")

        main_operator = self._qubit_converter.convert(
            main_second_q_op,
            num_particles=problem.num_particles,
            sector_locator=problem.symmetry_sector_locator,
        )
        aux_ops = self._qubit_converter.convert_match(aux_second_q_ops)

        if aux_operators is not None:
            for aux_op in aux_operators:
                if isinstance(aux_op, SecondQuantizedOp):
                    aux_ops.append(self._qubit_converter.convert_match(aux_op, True))
                else:
                    aux_ops.append(aux_op)

        if isinstance(self._solver, MinimumEigensolverFactory):
            # this must be called after transformation.transform
            self._solver = self._solver.get_solver(problem, self._qubit_converter)

        # if the eigensolver does not support auxiliary operators, reset them
        if not self._solver.supports_aux_operators():
            aux_ops = None

        raw_mes_result = self._solver.compute_minimum_eigenvalue(main_operator, aux_ops)

        result = problem.interpret(raw_mes_result)
        return result

    def evaluate_operators(
        self,
        state: Union[
            str,
            dict,
            Result,
            list,
            np.ndarray,
            Statevector,
            QuantumCircuit,
            Instruction,
            OperatorBase,
        ],
        operators: Union[PauliSumOp, OperatorBase, list, dict],
    ) -> Union[Optional[float], List[Optional[float]], Dict[str, List[Optional[float]]]]:
        """Evaluates additional operators at the given state.

        Args:
            state: any kind of input that can be used to specify a state. See also ``StateFn`` for
                   more details.
            operators: either a single, list or dictionary of ``PauliSumOp``s or any kind
                       of operator implementing the ``OperatorBase``.

        Returns:
            The expectation value of the given operator(s). The return type will be identical to the
            format of the provided operators.
        """
        # try to get a QuantumInstance from the solver
        quantum_instance = getattr(self._solver, "quantum_instance", None)
        # and try to get an Expectation from the solver
        expectation = getattr(self._solver, "expectation", None)

        if not isinstance(state, StateFn):
            state = StateFn(state)

        # handle all possible formats of operators
        # i.e. if a user gives us a dict of operators, we return the results equivalently, etc.
        if isinstance(operators, list):
            results = []  # type: ignore
            for op in operators:
                if op is None:
                    results.append(None)
                else:
                    results.append(self._eval_op(state, op, quantum_instance, expectation))
        elif isinstance(operators, dict):
            results = {}  # type: ignore
            for name, op in operators.items():
                if op is None:
                    results[name] = None
                else:
                    results[name] = self._eval_op(state, op, quantum_instance, expectation)
        else:
            if operators is None:
                results = None
            else:
                results = self._eval_op(state, operators, quantum_instance, expectation)

        return results

    def _eval_op(self, state, op, quantum_instance, expectation):
        # if the operator is empty we simply return 0
        if op == 0:
            # Note, that for some reason the individual results need to be wrapped in lists.
            # See also: VQE._eval_aux_ops()
            return [0.0j]

        exp = ~StateFn(op) @ state  # <state|op|state>

        if quantum_instance is not None:
            try:
                sampler = CircuitSampler(quantum_instance)
                if expectation is not None:
                    exp = expectation.convert(exp)
                result = sampler.convert(exp).eval()
            except ValueError:
                # TODO make this cleaner. The reason for it being here is that some quantum
                # instances can lead to non-positive statevectors which the Qiskit circuit
                # Initializer is unable to handle.
                result = exp.eval()
        else:
            result = exp.eval()

        # Note, that for some reason the individual results need to be wrapped in lists.
        # See also: VQE._eval_aux_ops()
        return [result]
