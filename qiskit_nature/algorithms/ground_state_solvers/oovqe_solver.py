# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The orbital optimization VQE solver"""

from typing import Union, List, Optional
import warnings
from time import time
import logging

from functools import partial

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.opflow import OperatorBase, PauliSumOp, StateFn
from qiskit.opflow.gradients import GradientBase
from qiskit.algorithms.minimum_eigen_solvers.vqe import (
    VQEResult,
    _validate_bounds,
    _validate_initial_point,
)
from qiskit.algorithms.minimum_eigen_solvers.minimum_eigen_solver import (
    MinimumEigensolverResult,
)

from qiskit_nature.converters.second_quantization.utils import ListOrDict

logger = logging.getLogger(__name__)


def energy_evaluation_oo(
    ground_state_eigensolver, solver, parameters: np.ndarray
) -> Union[float, List[float]]:
    """Doctstring"""
    num_parameters_ansatz = solver.ansatz.num_parameters
    if num_parameters_ansatz == 0:
        raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

    ansatz_params = solver.ansatz.parameters

    ansatz_parameter_values = parameters[:num_parameters_ansatz]
    rotation_parameter_values = parameters[num_parameters_ansatz:]  # is this the correct order?

    # CALCULATE COEFFICIENTS OF ROTATION MATRIX HERE:
    # matrix_a, matrix_b = np.eye(2), np.eye(2)
    matrix_a, matrix_b = ground_state_eigensolver.orbital_rotation.get_orbital_rotation_matrix(
        rotation_parameter_values
    )

    # ROTATE AND RECOMPUTE OPERATOR HERE:
    # what about aux_ops??? They should be rotated too.
    # Not implemented yet.

    rotated_operator = ground_state_eigensolver.rotate_orbitals(matrix_a, matrix_b)

    # use rotated operator for constructing expect_op
    expect_op, expectation = solver.construct_expectation(
        ansatz_params, rotated_operator, return_expectation=True
    )

    # the rest of the energy evaluation code only involves the ansatz parameters
    parameter_sets = np.reshape(ansatz_parameter_values, (-1, num_parameters_ansatz))

    # Create dict associating each parameter with the lists of parameterization values for it
    param_bindings = dict(zip(ansatz_params, parameter_sets.transpose().tolist()))

    start_time = time()
    sampled_expect_op = solver._circuit_sampler.convert(expect_op, params=param_bindings)
    means = np.real(sampled_expect_op.eval())

    if solver._callback is not None:
        variance = np.real(expectation.compute_variance(sampled_expect_op))
        estimator_error = np.sqrt(variance / solver.quantum_instance.run_config.shots)
        for i, param_set in enumerate(parameter_sets):
            solver._eval_count += 1
            solver._callback(solver._eval_count, param_set, means[i], estimator_error[i])
    else:
        solver._eval_count += len(means)

    end_time = time()
    logger.info(
        "Energy evaluation returned %s - %.5f (ms), eval count: %s",
        means,
        (end_time - start_time) * 1000,
        solver._eval_count,
    )

    return means if len(means) > 1 else means[0]


def compute_minimum_eigenvalue_oo(
    ground_state_eigensolver,
    solver,
    operator: OperatorBase,
    aux_operators: Optional[ListOrDict[OperatorBase]] = None,
) -> MinimumEigensolverResult:
    """Doctstring"""
    if solver.quantum_instance is None:
        raise QiskitError(
            "A QuantumInstance or Backend must be supplied to run the quantum algorithm."
        )
    solver.quantum_instance.circuit_summary = True

    # this sets the size of the ansatz, so it must be called before the initial point
    # validation
    solver._check_operator_ansatz(operator)

    # set an expectation for this algorithm run (will be reset to None at the end)
    initial_point_ansatz = _validate_initial_point(solver.initial_point, solver.ansatz)
    bounds_ansatz = _validate_bounds(solver.ansatz)

    # HERE: the real initial point and bounds include the ansatz and the oo parameters:
    bounds_oo_val: tuple = (-2 * np.pi, 2 * np.pi)
    initial_pt_scalar: float = 1e-1

    initial_point_oo = np.asarray(
        [initial_pt_scalar for _ in range(ground_state_eigensolver.orbital_rotation.num_parameters)]
    )
    bounds_oo = np.asarray(
        [bounds_oo_val for _ in range(ground_state_eigensolver.orbital_rotation.num_parameters)]
    )

    initial_point = np.concatenate((initial_point_ansatz, initial_point_oo))
    bounds = np.concatenate((bounds_ansatz, bounds_oo))

    # HERE: for the moment, not taking care of aux_operators
    # Does the orbital rotation affect them???
    # We need to handle the array entries being zero or Optional i.e. having value None
    if aux_operators:
        zero_op = PauliSumOp.from_list([("I" * solver.ansatz.num_qubits, 0)])

        # Convert the None and zero values when aux_operators is a list.
        # Drop None and convert zero values when aux_operators is a dict.
        if isinstance(aux_operators, list):
            key_op_iterator = enumerate(aux_operators)
            converted = [zero_op] * len(aux_operators)
        else:
            key_op_iterator = aux_operators.items()
            converted = {}
        for key, op in key_op_iterator:
            if op is not None:
                converted[key] = zero_op if op == 0 else op

        aux_operators = converted

    else:
        aux_operators = None

    # Convert the gradient operator into a callable function that is compatible with the
    # optimization routine.
    if isinstance(solver._gradient, GradientBase):
        gradient = solver._gradient.gradient_wrapper(
            ~StateFn(operator) @ StateFn(solver.ansatz),
            bind_params=list(solver.ansatz.parameters),
            backend=solver._quantum_instance,
        )
    else:
        gradient = solver._gradient

    solver._eval_count = 0

    # HERE: custom energy eval. function to pass to optimizer
    energy_evaluation = partial(energy_evaluation_oo, ground_state_eigensolver, solver)

    start_time = time()

    # keep this until Optimizer.optimize is removed
    try:
        opt_result = solver.optimizer.minimize(
            fun=energy_evaluation, x0=initial_point, jac=gradient, bounds=bounds
        )
    except AttributeError:
        # solver.optimizer is an optimizer with the deprecated interface that uses
        # ``optimize`` instead of ``minimize```
        warnings.warn(
            "Using an optimizer that is run with the ``optimize`` method is "
            "deprecated as of Qiskit Terra 0.19.0 and will be unsupported no "
            "sooner than 3 months after the release date. Instead use an optimizer "
            "providing ``minimize`` (see qiskit.algorithms.optimizers.Optimizer).",
            DeprecationWarning,
            stacklevel=2,
        )

        opt_result = solver.optimizer.optimize(
            len(initial_point), energy_evaluation, gradient, bounds, initial_point
        )

    eval_time = time() - start_time

    result = VQEResult()
    result.optimal_point = opt_result.x
    result.optimal_parameters = dict(zip(solver.ansatz.parameters, opt_result.x))
    result.optimal_value = opt_result.fun
    result.cost_function_evals = opt_result.nfev
    result.optimizer_time = eval_time
    result.eigenvalue = opt_result.fun + 0j
    result.eigenstate = solver._get_eigenstate(result.optimal_parameters)

    logger.info(
        "Optimization complete in %s seconds.\nFound opt_params %s in %s evals",
        eval_time,
        result.optimal_point,
        solver._eval_count,
    )

    # TODO delete as soon as get_optimal_vector etc are removed
    solver._ret = result

    if aux_operators is not None:
        # construct expectation AFTER optimization loop is finished, with rotated operator
        rotated_operator = operator
        # ADD ROTATION LOGIC HERE
        ansatz_params = solver.ansatz.parameters

        _, expectation = solver.construct_expectation(
            ansatz_params, rotated_operator, return_expectation=True
        )

        aux_values = solver._eval_aux_ops(
            opt_result.x[: solver.ansatz.num_parameters], aux_operators, expectation=expectation
        )
        result.aux_operator_eigenvalues = aux_values

    return result
