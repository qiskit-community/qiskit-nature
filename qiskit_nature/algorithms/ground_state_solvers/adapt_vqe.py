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

"""A ground state calculation employing the AdaptVQE algorithm."""

from typing import Optional, List, Tuple, Union

import copy
import re
import logging

import numpy as np

from qiskit.algorithms import VQE
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import OperatorBase, PauliSumOp, CircuitSampler
from qiskit.opflow.gradients import GradientBase, Gradient
from qiskit.utils.validation import validate_min
from qiskit_nature import ListOrDictType
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.circuit.library import UCC
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.converters.second_quantization.utils import ListOrDict
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.results import ElectronicStructureResult
from qiskit_nature.deprecation import deprecate_arguments

from .minimum_eigensolver_factories import MinimumEigensolverFactory
from .ground_state_eigensolver import GroundStateEigensolver

logger = logging.getLogger(__name__)


class AdaptVQE(GroundStateEigensolver):
    """A ground state calculation employing the AdaptVQE algorithm.

    The performance of AdaptVQE significantly depends on the choice of `gradient` (see also
    `qiskit.opflow.gradients`) and its parameters such as `grad_method`, `qfi_method` (if
    applicable) and `epilson`.

    To reproduce the default behavior of AdaptVQE prior to Qiskit Nature 0.4 you should supply
    `delta=1` explicitly. This will use a finite difference scheme for the gradient evaluation
    whereas after version 0.4 a parameter shift gradient will be used.
    For more information refer to the gradient framework of Qiskit Terra:
    https://qiskit.org/documentation/tutorials/operators/02_gradients_framework.html
    """

    @deprecate_arguments(
        "0.4.0",
        {"delta": "gradient"},
        additional_msg=(
            "Instead of `delta=1.0` you have to construct a gradient, like so "
            "`gradient=Gradient(grad_method='fin_diff', epsilon=1.0)`."
        ),
    )
    # pylint: disable=unused-argument
    def __init__(
        self,
        qubit_converter: QubitConverter,
        solver: MinimumEigensolverFactory,
        threshold: float = 1e-5,
        delta: float = 1.0,  # delta is copied into gradient by the deprecate_arguments wrapper
        max_iterations: Optional[int] = None,
        gradient: Optional[GradientBase] = None,
    ) -> None:
        """
        Args:
            qubit_converter: a class that converts second quantized operator to qubit operator
            solver: a factory for the VQE solver employing a UCCSD ansatz.
            threshold: the energy convergence threshold. It has a minimum value of 1e-15.
            delta: the finite difference step size for the gradient computation. It has a minimum
                value of 1e-5.
            max_iterations: the maximum number of iterations of the AdaptVQE algorithm.
            gradient: a class that converts operator expression to the first-order gradient based
                on the method mentioned.
        """
        validate_min("threshold", threshold, 1e-15)

        if isinstance(gradient, float):
            # this scenario can only occur while using the deprecate_arguments wrapper which will
            # move any argument supplied to delta into gradient.
            gradient = Gradient(grad_method="fin_diff", epsilon=gradient)

        super().__init__(qubit_converter, solver)

        self._threshold = threshold
        self._max_iterations = max_iterations
        self.gradient = gradient  # use the property setter to handle `None` case consistently

        self._excitation_pool: List[OperatorBase] = []
        self._excitation_list: List[OperatorBase] = []

        self._main_operator: PauliSumOp = None
        self._ansatz: QuantumCircuit = None
        self._sampler: CircuitSampler = None

    @property
    def gradient(self) -> GradientBase:
        """Returns the gradient."""
        return self._gradient

    @gradient.setter
    def gradient(self, grad: Optional[GradientBase] = None) -> None:
        """Sets the gradient."""
        if grad is None:
            grad = Gradient(grad_method="param_shift")
        self._gradient = grad

    def returns_groundstate(self) -> bool:
        """Whether this class returns only the ground state energy or also the ground state itself."""
        return True

    def _compute_gradients(
        self,
        theta: List[float],
        vqe: VQE,
    ) -> List[Tuple[float, PauliSumOp]]:
        """
        Computes the gradients for all available excitation operators.

        Args:
            theta: list of (up to now) optimal parameters
            vqe: the variational quantum eigensolver instance used for solving

        Returns:
            List of pairs consisting of gradient and excitation operator.
        """
        res = []
        # compute gradients for all excitation in operator pool
        for exc in self._excitation_pool:
            # add next excitation to ansatz
            self._ansatz.operators = self._excitation_list + [exc]
            # Due to an outstanding issue in Terra, the ansatz needs to be decomposed for all
            # gradient to work correctly. Once this issue is resolved, this workaround can be
            # removed.
            vqe.ansatz = self._ansatz.decompose()
            # We need to explicitly convert this to a list in order to make the object hash-able
            param_sets = list(vqe.ansatz.parameters)
            # zip will only iterate the length of the shorter list
            theta1 = dict(zip(vqe.ansatz.parameters, theta))
            op = vqe.construct_expectation(theta1, self._main_operator)
            # compute gradient
            state_grad = self.gradient.convert(operator=op, params=param_sets)
            # Assign the parameters and evaluate the gradient
            value_dict = {param_sets[-1]: 0.0}
            state_grad_result = self._sampler.convert(state_grad, params=value_dict).eval()
            logger.info("Gradient computed : %s", str(state_grad_result))
            res.append((np.abs(state_grad_result[-1]), exc))
        return res

    @staticmethod
    def _check_cyclicity(indices: List[int]) -> bool:
        """
        Auxiliary function to check for cycles in the indices of the selected excitations.

        Args:
            indices: the list of chosen gradient indices.
        Returns:
            Whether repeating sequences of indices have been detected.
        """
        cycle_regex = re.compile(r"(\b.+ .+\b)( \b\1\b)+")
        # reg-ex explanation:
        # 1. (\b.+ .+\b) will match at least two numbers and try to match as many as possible. The
        #    word boundaries in the beginning and end ensure that now numbers are split into digits.
        # 2. the match of this part is placed into capture group 1
        # 3. ( \b\1\b)+ will match a space followed by the contents of capture group 1 (again
        #    delimited by word boundaries to avoid separation into digits).
        # -> this results in any sequence of at least two numbers being detected
        match = cycle_regex.search(" ".join(map(str, indices)))
        logger.debug("Cycle detected: %s", match)
        # Additionally we also need to check whether the last two numbers are identical, because the
        # reg-ex above will only find cycles of at least two consecutive numbers.
        # It is sufficient to assert that the last two numbers are different due to the iterative
        # nature of the algorithm.
        return match is not None or (len(indices) > 1 and indices[-2] == indices[-1])

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]] = None,
    ) -> "AdaptVQEResult":
        """Computes the ground state.

        Args:
            problem: a class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Raises:
            QiskitNatureError: if a solver other than VQE or a ansatz other than UCCSD is provided
                or if the algorithm finishes due to an unforeseen reason.
            ValueError: if the grouped property object returned by the driver does not contain a
                main property as requested by the problem being solved (`problem.main_property_name`)
            QiskitNatureError: if the user-provided `aux_operators` contain a name which clashes
                with an internally constructed auxiliary operator. Note: the names used for the
                internal auxiliary operators correspond to the `Property.name` attributes which
                generated the respective operators.
            QiskitNatureError: if the chosen gradient method appears to result in all-zero gradients.

        Returns:
            An AdaptVQEResult which is an ElectronicStructureResult but also includes runtime
            information about the AdaptVQE algorithm like the number of iterations, finishing
            criterion, and the final maximum gradient.
        """
        second_q_ops = problem.second_q_ops()

        aux_second_q_ops: ListOrDictType[SecondQuantizedOp]
        if isinstance(second_q_ops, list):
            main_second_q_op = second_q_ops[0]
            aux_second_q_ops = second_q_ops[1:]
        elif isinstance(second_q_ops, dict):
            name = problem.main_property_name
            main_second_q_op = second_q_ops.pop(name, None)
            if main_second_q_op is None:
                raise ValueError(
                    f"The main `SecondQuantizedOp` associated with the {name} property cannot be "
                    "`None`."
                )
            aux_second_q_ops = second_q_ops

        self._main_operator = self._qubit_converter.convert(
            main_second_q_op,
            num_particles=problem.num_particles,
            sector_locator=problem.symmetry_sector_locator,
        )
        aux_ops = self._qubit_converter.convert_match(aux_second_q_ops)

        if aux_operators is not None:
            wrapped_aux_operators: ListOrDict[Union[SecondQuantizedOp, PauliSumOp]] = ListOrDict(
                aux_operators
            )
            for name_aux, aux_op in iter(wrapped_aux_operators):
                if isinstance(aux_op, SecondQuantizedOp):
                    converted_aux_op = self._qubit_converter.convert_match(aux_op, True)
                else:
                    converted_aux_op = aux_op
                if isinstance(aux_ops, list):
                    aux_ops.append(converted_aux_op)
                elif isinstance(aux_ops, dict):
                    if name_aux in aux_ops.keys():
                        raise QiskitNatureError(
                            f"The key '{name_aux}' is already taken by an internally constructed "
                            "auxiliary operator! Please use a different name for your custom "
                            "operator."
                        )
                    aux_ops[name_aux] = converted_aux_op

        if isinstance(self._solver, MinimumEigensolverFactory):
            vqe = self._solver.get_solver(problem, self._qubit_converter)
        else:
            vqe = self._solver

        if not isinstance(vqe, VQE):
            raise QiskitNatureError("The AdaptVQE algorithm requires the use of the VQE solver")
        if not isinstance(vqe.ansatz, UCC):
            raise QiskitNatureError("The AdaptVQE algorithm requires the use of the UCC ansatz")

        # setup circuit sampler
        self._sampler = CircuitSampler(vqe.quantum_instance)

        # We construct the ansatz once to be able to extract the full set of excitation operators.
        self._ansatz = copy.deepcopy(vqe.ansatz)
        self._ansatz._build()
        self._excitation_pool = copy.deepcopy(self._ansatz.operators)

        threshold_satisfied = False
        alternating_sequence = False
        max_iterations_exceeded = False
        prev_op_indices: List[int] = []
        theta: List[float] = []
        max_grad: Tuple[float, Optional[PauliSumOp]] = (0.0, None)
        iteration = 0
        while self._max_iterations is None or iteration < self._max_iterations:
            iteration += 1
            logger.info("--- Iteration #%s ---", str(iteration))
            # compute gradients

            cur_grads = self._compute_gradients(theta, vqe)
            # pick maximum gradient
            max_grad_index, max_grad = max(
                enumerate(cur_grads), key=lambda item: np.abs(item[1][0])
            )
            # store maximum gradient's index for cycle detection
            prev_op_indices.append(max_grad_index)
            # log gradients
            if logger.isEnabledFor(logging.INFO):
                gradlog = f"\nGradients in iteration #{str(iteration)}"
                gradlog += "\nID: Excitation Operator: Gradient  <(*) maximum>"
                for i, grad in enumerate(cur_grads):
                    gradlog += f"\n{str(i)}: {str(grad[1])}: {str(grad[0])}"
                    if grad[1] == max_grad[1]:
                        gradlog += "\t(*)"
                logger.info(gradlog)
            if np.abs(max_grad[0]) < self._threshold:
                if iteration == 1:
                    raise QiskitNatureError(
                        "Gradient choice is not suited as it leads to all zero gradients gradients. "
                        "Try a different gradient method."
                    )
                logger.info(
                    "Adaptive VQE terminated successfully with a final maximum gradient: %s",
                    str(np.abs(max_grad[0])),
                )
                threshold_satisfied = True
                break
            # check indices of picked gradients for cycles
            if self._check_cyclicity(prev_op_indices):
                logger.info("Alternating sequence found. Finishing.")
                logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))
                alternating_sequence = True
                break
            # add new excitation to self._ansatz
            self._excitation_list.append(max_grad[1])
            theta.append(0.0)
            # run VQE on current Ansatz
            self._ansatz.operators = self._excitation_list
            vqe.ansatz = self._ansatz
            vqe.initial_point = theta
            raw_vqe_result = vqe.compute_minimum_eigenvalue(self._main_operator)
            theta = raw_vqe_result.optimal_point.tolist()
        else:
            # reached maximum number of iterations
            max_iterations_exceeded = True
            logger.info("Maximum number of iterations reached. Finishing.")
            logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))

        # once finished evaluate auxiliary operators if any
        if aux_ops is not None:
            aux_values = self.evaluate_operators(raw_vqe_result.eigenstate, aux_ops)
        else:
            aux_values = None
        raw_vqe_result.aux_operator_eigenvalues = aux_values

        if threshold_satisfied:
            finishing_criterion = "Threshold converged"
        elif alternating_sequence:
            finishing_criterion = "Aborted due to cyclicity"
        elif max_iterations_exceeded:
            finishing_criterion = "Maximum number of iterations reached"
        else:
            raise QiskitNatureError("The algorithm finished due to an unforeseen reason!")

        electronic_result = problem.interpret(raw_vqe_result)

        result = AdaptVQEResult()
        result.combine(electronic_result)
        result.num_iterations = iteration
        result.final_max_gradient = max_grad[0]
        result.finishing_criterion = finishing_criterion

        logger.info("The final energy is: %s", str(result.computed_energies[0]))
        return result


class AdaptVQEResult(ElectronicStructureResult):
    """AdaptVQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._num_iterations: int = 0
        self._final_max_gradient: float = 0.0
        self._finishing_criterion: str = ""

    @property
    def num_iterations(self) -> int:
        """Returns number of iterations"""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value: int) -> None:
        """Sets number of iterations"""
        self._num_iterations = value

    @property
    def final_max_gradient(self) -> float:
        """Returns final maximum gradient"""
        return self._final_max_gradient

    @final_max_gradient.setter
    def final_max_gradient(self, value: float) -> None:
        """Sets final maximum gradient"""
        self._final_max_gradient = value

    @property
    def finishing_criterion(self) -> str:
        """Returns finishing criterion"""
        return self._finishing_criterion

    @finishing_criterion.setter
    def finishing_criterion(self, value: str) -> None:
        """Sets finishing criterion"""
        self._finishing_criterion = value
