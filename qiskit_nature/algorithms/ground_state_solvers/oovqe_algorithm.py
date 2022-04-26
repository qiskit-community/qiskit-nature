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

"""The orbital optimization VQE algorithm"""

from typing import Union, List, Optional
import warnings
import copy
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
    MinimumEigensolver,
    MinimumEigensolverResult,
)

from qiskit_nature import ListOrDictType, QiskitNatureError
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.converters.second_quantization.utils import ListOrDict
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.results import EigenstateResult

from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import (
    MinimumEigensolverFactory,
)
from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)

from qiskit_nature.algorithms.ground_state_solvers.custom_problem import CustomProblem
from qiskit_nature.algorithms.ground_state_solvers.orbital_rotation import OrbitalRotation

logger = logging.getLogger(__name__)

class OrbitalOptimizationVQE(GroundStateEigensolver):
    """Solver for ooVQE"""

    def __init__(
        self,
        qubit_converter: QubitConverter,
        solver: Union[MinimumEigensolver, MinimumEigensolverFactory],
    ) -> None:
        super().__init__(qubit_converter, solver)

        # Store problem to have access during energy eval. function.
        self.problem: CustomProblem = (
            None  # I am using temporarily the CustomProblem class, that avoids
        )
        # running the driver every time .second_q_ops() is called

        self.initial_point = None  # in the future: set by user
        self.bounds_oo: np.array = None  # in the future: set by user
        self.bounds: np.array = None  # ansatz + oo

        self.orbital_rotation = OrbitalRotation(
            num_qubits=self.solver.ansatz.num_qubits, qubit_converter=qubit_converter
        )
        self.num_parameters_oovqe = (
            self.solver.ansatz.num_parameters + self.orbital_rotation.num_parameters
        )

        # the initial point of the full ooVQE alg.
        if self.initial_point is None:
            self.set_initial_point()
        else:
            # this will never really happen with the current code
            # but is kept for the future
            if len(self.initial_point) is not self.num_parameters_oovqe:
                raise QiskitNatureError(
                    f"Number of parameters of OOVQE ({self.num_parameters_oovqe,}) "
                    f"does not match the length of the "
                    f"intitial_point ({len(self.initial_point)})"
                )

        if self.bounds is None:
            # set bounds sets both ansatz and oo bounds
            # do we want to change the ansatz bounds here??
            self.set_bounds(self.orbital_rotation.parameter_bound_value)

    def set_initial_point(self, initial_pt_scalar: float = 1e-1) -> None:
        """Initializes the initial point for the algorithm if the user does not provide his own.
        Args:
            initial_pt_scalar: value of the initial parameters for wavefunction and orbital rotation
        """
        self.initial_point = np.asarray(
            [initial_pt_scalar for _ in range(self.num_parameters_oovqe)]
        )

    def set_bounds(
        self,
        bounds_ansatz_value: tuple = (-2 * np.pi, 2 * np.pi),
        bounds_oo_value: tuple = (-2 * np.pi, 2 * np.pi),
    ) -> None:
        """Doctstring"""
        bounds_ansatz = [bounds_ansatz_value for _ in range(self.solver.ansatz.num_parameters)]
        self.bounds_oo = [bounds_oo_value for _ in range(self.orbital_rotation.num_parameters)]
        bounds = bounds_ansatz + self.bounds_oo
        self.bounds = np.array(bounds)

    def get_operators(self, problem, aux_operators):
        """Doctstring"""
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

        main_operator = self._qubit_converter.convert(
            main_second_q_op,
            num_particles=problem.num_particles,
            sector_locator=problem.symmetry_sector_locator,
        )
        aux_ops = self._qubit_converter.convert_match(aux_second_q_ops)

        if aux_operators is not None:
            wrapped_aux_operators: ListOrDict[Union[SecondQuantizedOp, PauliSumOp]] = ListOrDict(
                aux_operators
            )
            for name, aux_op in iter(wrapped_aux_operators):
                if isinstance(aux_op, SecondQuantizedOp):
                    converted_aux_op = self._qubit_converter.convert_match(aux_op, True)
                else:
                    converted_aux_op = aux_op
                if isinstance(aux_ops, list):
                    aux_ops.append(converted_aux_op)
                elif isinstance(aux_ops, dict):
                    if name in aux_ops.keys():
                        raise QiskitNatureError(
                            f"The key '{name}' is already taken by an internally constructed "
                            "auxliliary operator! Please use a different name for your custom "
                            "operator."
                        )
                    aux_ops[name] = converted_aux_op

        # if the eigensolver does not support auxiliary operators, reset them
        if not self._solver.supports_aux_operators():
            aux_ops = None

        return main_operator, aux_ops

    def rotate_orbitals(self, matrix_a, matrix_b):
        """Doctstring"""
        problem = copy.copy(self.problem)
        grouped_property_transformed = problem.grouped_property_transformed

        # use ElectronicBasisTransform
        transform = ElectronicBasisTransform(
            ElectronicBasis.MO, ElectronicBasis.MO, matrix_a, matrix_b
        )

        # only 1 & 2 body integrals have the "transform_basis" method,
        # so I access them through the electronic energy
        e_energy = grouped_property_transformed.get_property("ElectronicEnergy")
        one_body_integrals = e_energy.get_electronic_integral(ElectronicBasis.MO, 1)
        two_body_integrals = e_energy.get_electronic_integral(ElectronicBasis.MO, 2)

        # the basis transform should be applied in place, but it's not???
        # unless I manually add the integrals, the result of second_q_ops()
        # doesn't change.
        # I have to look further into this.
        e_energy.add_electronic_integral(one_body_integrals.transform_basis(transform))
        e_energy.add_electronic_integral(two_body_integrals.transform_basis(transform))

        # after applying the rotation, recompute operator
        rotated_main_second_q_op = e_energy.second_q_ops()
        rotated_operator = self._qubit_converter.convert(
            rotated_main_second_q_op,
            num_particles=problem.num_particles,
            sector_locator=problem.symmetry_sector_locator,
        )
        return rotated_operator

    def energy_evaluation_oo(self, solver, parameters: np.ndarray) -> Union[float, List[float]]:
        """Doctstring"""
        num_parameters_ansatz = solver.ansatz.num_parameters
        if num_parameters_ansatz == 0:
            raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

        ansatz_params = solver.ansatz.parameters

        ansatz_parameter_values = parameters[:num_parameters_ansatz]
        rotation_parameter_values = parameters[num_parameters_ansatz:]  # is this the correct order?

        # CALCULATE COEFFICIENTS OF ROTATION MATRIX HERE:
        # matrix_a, matrix_b = np.eye(2), np.eye(2)
        matrix_a, matrix_b = self.orbital_rotation.get_orbital_rotation_matrix(
            rotation_parameter_values
        )

        # ROTATE AND RECOMPUTE OPERATOR HERE:
        # what about aux_ops??? They should be rotated too.
        # Not implemented yet.
        rotated_operator = self.rotate_orbitals(matrix_a, matrix_b)

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
        self,
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
            [initial_pt_scalar for _ in range(self.orbital_rotation.num_parameters)]
        )
        bounds_oo = np.asarray([bounds_oo_val for _ in range(self.orbital_rotation.num_parameters)])

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
        energy_evaluation = partial(self.energy_evaluation_oo, solver)

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

            aux_values = solver._eval_aux_ops(opt_result.x, aux_operators, expectation=expectation)
            result.aux_operator_eigenvalues = aux_values

        return result

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]] = None,
    ) -> EigenstateResult:
        """Compute Ground State properties.

        Args:
            problem: a class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Raises:
            ValueError: if the grouped property object returned by the driver does not contain a
                main property as requested by the problem being solved (`problem.main_property_name`)
            QiskitNatureError: if the user-provided `aux_operators` contain a name which clashes
                with an internally constructed auxiliary operator. Note: the names used for the
                internal auxiliary operators correspond to the `Property.name` attributes which
                generated the respective operators.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """
        # get the operator and auxiliary operators, and transform the provided auxiliary operators
        # note that ``aux_ops`` contains not only the transformed ``aux_operators`` passed by the
        # user but also additional ones from the transformation

        self.problem = problem

        # override VQE's compute_minimum_eigenvalue, giving it access to the problem data
        # contained in self.problem
        self.solver.compute_minimum_eigenvalue = partial(
            self.compute_minimum_eigenvalue_oo, self.solver
        )

        main_operator, aux_ops = self.get_operators(problem, aux_operators)
        raw_mes_result = self.solver.compute_minimum_eigenvalue(main_operator, aux_ops)

        result = problem.interpret(raw_mes_result)

        return result
