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

"""The calculation of points on the Born-Oppenheimer Potential Energy Surface (BOPES)."""

import logging
from typing import Optional, List, Dict, Union, Callable

import numpy as np

from qiskit.algorithms import VariationalAlgorithm
from qiskit.opflow import PauliSumOp
from qiskit.utils.deprecation import deprecate_arguments

from qiskit_nature.drivers.second_quantization import (
    BaseDriver,
    ElectronicStructureMoleculeDriver,
    VibrationalStructureMoleculeDriver,
)
from qiskit_nature import ListOrDictType
from qiskit_nature.converters.second_quantization.utils import ListOrDict
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_quantization.operators import SecondQuantizedOp
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.second_quantization.results import BOPESSamplerResult, EigenstateResult
from .extrapolator import Extrapolator, WindowExtrapolator
from ..ground_state_solvers import GroundStateSolver
from ..excited_states_solvers import ExcitedStatesSolver

logger = logging.getLogger(__name__)


class BOPESSampler:
    """Class to evaluate the Born-Oppenheimer Potential Energy Surface (BOPES)."""

    @deprecate_arguments({"gss": "state_solver"})
    # pylint: disable=unused-argument
    def __init__(
        self,
        state_solver: Optional[Union[GroundStateSolver, ExcitedStatesSolver]] = None,
        tolerance: float = 1e-3,
        bootstrap: bool = True,
        num_bootstrap: Optional[int] = None,
        extrapolator: Optional[Extrapolator] = None,
        gss: Optional[GroundStateSolver] = None,
    ) -> None:
        """
        Args:
            state_solver: GroundStateSolver or ExcitedStatesSolver.
            tolerance: Tolerance desired for minimum energy.
            bootstrap: Whether to warm-start the solution of variational minimum eigensolvers.
            num_bootstrap: Number of previous points for extrapolation
                and bootstrapping. If None and a list of extrapolators is defined,
                the first two points will be used for bootstrapping.
                If no extrapolator is defined and bootstrap is True,
                all previous points will be used for bootstrapping.
            extrapolator: Extrapolator objects that define space/window
                           and method to extrapolate variational parameters.
            gss: (DEPRECATED) Old name for the solver when it only supported GroundStateSolver.
                Note that gss is copied into state_solver by the deprecate_arguments wrapper.

        Raises:
            QiskitNatureError: If ``num_boostrap`` is an integer smaller than 2, or
                if ``num_boostrap`` is larger than 2 and the extrapolator is not an instance of
                ``WindowExtrapolator``.
        """

        self._state_solver = state_solver
        self._is_variational_solver: bool = False
        self._tolerance = tolerance
        self._bootstrap = bootstrap
        self._aux_operators = None
        self._problem: BaseProblem = None
        self._driver: BaseDriver = None
        self._points: List[float] = None
        self._energies: List[List[float]] = None
        self._raw_results: Dict[float, EigenstateResult] = None
        self._points_optparams: Dict[float, List[float]] = None
        self._num_bootstrap = num_bootstrap
        self._extrapolator = extrapolator

        if self._extrapolator:
            if num_bootstrap is None:
                # set default number of bootstrapping points to 2
                self._num_bootstrap = 2
            elif num_bootstrap >= 2:
                if not isinstance(self._extrapolator, WindowExtrapolator):
                    raise QiskitNatureError(
                        "If num_bootstrap >= 2 then the extrapolator must be an instance "
                        f"of WindowExtrapolator, got {self._extrapolator} instead"
                    )
                self._num_bootstrap = num_bootstrap
                self._extrapolator.window = num_bootstrap  # window for extrapolator
            else:
                raise QiskitNatureError(
                    "num_bootstrap must be None or an integer greater than or equal to 2"
                )
        if self._is_variational_solver:
            # Save initial point passed to min_eigensolver;
            # this will be used when NOT bootstrapping
            self._initial_point = self._state_solver.solver.initial_point

    def sample(
        self,
        problem: BaseProblem,
        points: List[float],
        aux_operators: Optional[
            ListOrDictType[
                Union[
                    SecondQuantizedOp,
                    PauliSumOp,
                    Callable[[BaseProblem], Union[SecondQuantizedOp, PauliSumOp]],
                ]
            ]
        ] = None,
    ) -> BOPESSamplerResult:
        """Run the sampler at the given points, potentially with repetitions.
        Auxiliary operators can be evaluated at each step.
        If they are given as operators, the given operator will be evaluated at each step of the
        BOPESSampler.
        The signature also allows `Callable` object to define geometry dependent operators.
        The `evaluate_callable_aux_operators()` will convert all the `Callable` object of the input
        `aux_operators` to their corresponding auxiliaries at the current step.
        The resulting ListOrDict of aux_operators will then be passed to the solver.
        `Callable` auxiliary operators should access the properties of the current step through a
        `BaseProblem` argument and return a valid auxiliary operator.

        Args:
            problem: BaseProblem whose driver should be based on a Molecule object that has
                     perturbations to be varied.
            points: The points along the degrees of freedom to evaluate.
            aux_operators: Auxiliary operators to pass to the state_solver object.

        Returns:
            BOPES Sampler Result

        Raises:
            QiskitNatureError: if the driver does not have a molecule specified.
        """
        self._problem = problem
        self._driver = problem.driver
        self._aux_operators = aux_operators
        # We have to force the creation of the solver so that we work on the same solver
        # instance before and after `_state_solver.solve`.
        self._state_solver.get_qubit_operators(problem=problem, aux_operators=None)
        # this must be called after self._state_solver.get_qubit_operators to account for
        # EigenSolverFactories.
        self._is_variational_solver = isinstance(self._state_solver.solver, VariationalAlgorithm)

        if self._is_variational_solver:
            # Save initial point passed to min_eigensolver;
            # this will be used when NOT bootstrapping
            self._initial_point = self._state_solver.solver.initial_point

        if not isinstance(
            self._driver, (ElectronicStructureMoleculeDriver, VibrationalStructureMoleculeDriver)
        ):
            raise QiskitNatureError(
                "Driver must be ElectronicStructureMoleculeDriver or VibrationalStructureMoleculeDriver."
            )

        if self._driver.molecule is None:
            raise QiskitNatureError("Driver MUST be configured with a Molecule.")

        # full dictionary of points
        self._raw_results = self._run_points(points)
        # create results dictionary with (point, energy)
        self._points = list(self._raw_results.keys())
        self._energies = [res.total_energies for res in self._raw_results.values()]

        result = BOPESSamplerResult(self._points, self._energies, self._raw_results)

        return result

    def _run_points(self, points: List[float]) -> Dict[float, EigenstateResult]:
        """Run the sampler at the given points.

        Args:
            points: the points along the single degree of freedom to evaluate

        Returns:
            The results for all points.
        """
        raw_results: Dict[float, EigenstateResult] = {}
        if self._is_variational_solver:
            self._points_optparams = {}
            self._state_solver.solver.initial_point = self._initial_point

        # Iterate over the points
        for i, point in enumerate(points):
            logger.info("Point %s of %s", i + 1, len(points))
            raw_result = self._run_single_point(point)  # dict of results
            raw_results[point] = raw_result

        return raw_results

    def _run_single_point(self, point: float) -> EigenstateResult:
        """Run the sampler at the given single point

        Args:
            point: The value of the degree of freedom to evaluate.

        Returns:
            Results for a single point.
        Raises:
            QiskitNatureError: Invalid Driver
        """

        # update molecule geometry and thus resulting Hamiltonian based on specified point

        if not isinstance(
            self._driver, (ElectronicStructureMoleculeDriver, VibrationalStructureMoleculeDriver)
        ):
            raise QiskitNatureError(
                "Driver must be ElectronicStructureMoleculeDriver or VibrationalStructureMoleculeDriver."
            )

        self._driver.molecule.perturbations = [point]

        # find closest previously run point and take optimal parameters
        if self._bootstrap and self._is_variational_solver:
            prev_points = list(self._points_optparams.keys())
            prev_params = list(self._points_optparams.values())
            n_pp = len(prev_points)

            # set number of points to bootstrap
            if self._extrapolator is None:
                n_boot = len(prev_points)  # bootstrap all points
            else:
                n_boot = self._num_bootstrap

            # Set initial params # if prev_points not empty
            if prev_points:
                if n_pp <= n_boot:
                    distances = np.array(point) - np.array(prev_points).reshape(n_pp, -1)
                    # find min 'distance' from point to previous points
                    min_index = int(np.argmin(np.linalg.norm(distances, axis=1)))
                    # update initial point
                    self._state_solver.solver.initial_point = prev_params[min_index]
                else:  # extrapolate using saved parameters
                    opt_params = self._points_optparams
                    param_sets = self._extrapolator.extrapolate(
                        points=[point], param_dict=opt_params
                    )
                    # update initial point, note param_set is a dictionary
                    self._state_solver.solver.initial_point = param_sets.get(point)

        # the output is an instance of EigenstateResult
        aux_ops_current_step = self.evaluate_callable_aux_operators()
        result = self._state_solver.solve(self._problem, aux_ops_current_step)

        # Save optimal point to bootstrap
        if self._is_variational_solver:
            if isinstance(self._state_solver, ExcitedStatesSolver):
                optimal_params = result.raw_result.ground_state_raw_result.optimal_point
            elif isinstance(self._state_solver, GroundStateSolver):
                optimal_params = result.raw_result.optimal_point
            self._points_optparams[point] = optimal_params

        return result

    def evaluate_callable_aux_operators(
        self,
    ) -> ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]:
        """Convert the dictionary of auxiliary observables stored in self at the beginning of the
        sample() method into a dictionary of auxiliary observables where the possible `Callable`
        observables have been evaluated for the current step.
        For example, this can be used to specify nuclear coordinate dependent observables.
        """

        aux_ops_current_step = None
        if self._aux_operators is not None:
            aux_ops_current_step = ListOrDict()
            for aux_name, aux_op in iter(ListOrDict(self._aux_operators)):
                if callable(aux_op):
                    aux_ops_current_step[aux_name] = aux_op(self._problem)
                else:
                    aux_ops_current_step[aux_name] = aux_op
        return aux_ops_current_step
