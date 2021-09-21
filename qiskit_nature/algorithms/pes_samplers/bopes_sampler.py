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

"""The calculation of points on the Born-Oppenheimer Potential Energy Surface (BOPES)."""

import logging
from typing import Optional, List, Dict, Union

import numpy as np
from qiskit.algorithms import VariationalAlgorithm
from qiskit_nature.deprecation import DeprecatedType, warn_deprecated
from qiskit_nature.drivers.base_driver import BaseDriver as DeprecatedBaseDriver
from qiskit_nature.drivers.second_quantization import (
    BaseDriver,
    ElectronicStructureMoleculeDriver,
    VibrationalStructureMoleculeDriver,
)
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.results import BOPESSamplerResult, EigenstateResult
from ..ground_state_solvers import GroundStateSolver
from .extrapolator import Extrapolator, WindowExtrapolator

logger = logging.getLogger(__name__)


class BOPESSampler:
    """Class to evaluate the Born-Oppenheimer Potential Energy Surface (BOPES)."""

    def __init__(
        self,
        gss: GroundStateSolver,
        tolerance: float = 1e-3,
        bootstrap: bool = True,
        num_bootstrap: Optional[int] = None,
        extrapolator: Optional[Extrapolator] = None,
    ) -> None:
        """
        Args:
            gss: GroundStateSolver
            tolerance: Tolerance desired for minimum energy.
            bootstrap: Whether to warm-start the solution of variational minimum eigensolvers.
            num_bootstrap: Number of previous points for extrapolation
                and bootstrapping. If None and a list of extrapolators is defined,
                the first two points will be used for bootstrapping.
                If no extrapolator is defined and bootstrap is True,
                all previous points will be used for bootstrapping.
            extrapolator: Extrapolator objects that define space/window
                           and method to extrapolate variational parameters.

        Raises:
            QiskitNatureError: If ``num_boostrap`` is an integer smaller than 2, or
                if ``num_boostrap`` is larger than 2 and the extrapolator is not an instance of
                ``WindowExtrapolator``.
        """

        self._gss = gss
        self._tolerance = tolerance
        self._bootstrap = bootstrap
        self._problem: BaseProblem = None
        self._driver: Union[DeprecatedBaseDriver, BaseDriver] = None
        self._points: List[float] = None
        self._energies: List[float] = None
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

        if isinstance(self._gss.solver, VariationalAlgorithm):  # type: ignore
            # Save initial point passed to min_eigensolver;
            # this will be used when NOT bootstrapping
            self._initial_point = self._gss.solver.initial_point  # type: ignore

    def sample(self, problem: BaseProblem, points: List[float]) -> BOPESSamplerResult:
        """Run the sampler at the given points, potentially with repetitions.

        Args:
            problem: BaseProblem whose driver should be based on a Molecule object that has
                     perturbations to be varied.
            points: The points along the degrees of freedom to evaluate.

        Returns:
            BOPES Sampler Result

        Raises:
            QiskitNatureError: if the driver does not have a molecule specified.
        """
        self._problem = problem
        self._driver = problem.driver

        if isinstance(self._driver, DeprecatedBaseDriver):
            warn_deprecated(
                "0.2.0",
                DeprecatedType.CLASS,
                f"{self._driver.__class__.__module__}.{self._driver.__class__.__qualname__}",
                new_name="ElectronicStructureMoleculeDriver or VibrationalStructureMoleculeDriver",
                additional_msg="from qiskit_nature.drivers.second_quantization",
            )
        elif not isinstance(
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
        self._energies = []
        for res in self._raw_results.values():
            energy = res.total_energies[0]
            self._energies.append(energy)

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
        if isinstance(self._gss.solver, VariationalAlgorithm):  # type: ignore
            self._points_optparams = {}
            self._gss.solver.initial_point = self._initial_point  # type: ignore

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

        if isinstance(self._driver, DeprecatedBaseDriver):
            warn_deprecated(
                "0.2.0",
                DeprecatedType.CLASS,
                f"{self._driver.__class__.__module__}.{self._driver.__class__.__qualname__}",
                new_name="ElectronicStructureMoleculeDriver or VibrationalStructureMoleculeDriver",
                additional_msg="from qiskit_nature.drivers.second_quantization",
            )
        elif not isinstance(
            self._driver, (ElectronicStructureMoleculeDriver, VibrationalStructureMoleculeDriver)
        ):
            raise QiskitNatureError(
                "Driver must be ElectronicStructureMoleculeDriver or VibrationalStructureMoleculeDriver."
            )

        self._driver.molecule.perturbations = [point]

        # find closest previously run point and take optimal parameters
        if isinstance(self._gss.solver, VariationalAlgorithm) and self._bootstrap:  # type: ignore
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
                    min_index = np.argmin(np.linalg.norm(distances, axis=1))
                    # update initial point
                    self._gss.solver.initial_point = prev_params[min_index]  # type: ignore
                else:  # extrapolate using saved parameters
                    opt_params = self._points_optparams
                    param_sets = self._extrapolator.extrapolate(
                        points=[point], param_dict=opt_params
                    )
                    # update initial point, note param_set is a dictionary
                    self._gss.solver.initial_point = param_sets.get(point)  # type: ignore

        # the output is an instance of EigenstateResult
        result = self._gss.solve(self._problem)

        # Save optimal point to bootstrap
        if isinstance(self._gss.solver, VariationalAlgorithm):  # type: ignore
            # at every point evaluation, the optimal params are updated
            optimal_params = result.raw_result.optimal_point  # type: ignore
            self._points_optparams[point] = optimal_params

        return result
