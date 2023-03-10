# This code is part of Qiskit.
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

"""The minimum eigensolver factory for ground state calculation algorithms."""

from __future__ import annotations

import logging
import numpy as np
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolver, VQE
from qiskit.algorithms.optimizers import Minimizer, Optimizer
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimator

from qiskit_nature.deprecation import DeprecatedType, warn_deprecated
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.mappers import QubitConverter, QubitMapper
from qiskit_nature.second_q.problems import (
    ElectronicStructureProblem,
)
from qiskit_nature.deprecation import deprecate_arguments

from ...initial_points import InitialPoint, HFInitialPoint
from .minimum_eigensolver_factory import MinimumEigensolverFactory

logger = logging.getLogger(__name__)


class VQEUCCFactory(MinimumEigensolverFactory):
    """DEPRECATED Factory to construct a :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`
    minimum eigensolver with :class:`~.UCC` ansatz wavefunction.

    .. warning::

        This class is deprecated! Please see :ref:`this guide <how-to-vqe-ucc>` for how to replace
        your usage of it!
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: UCC,
        optimizer: Optimizer | Minimizer,
        *,
        initial_point: np.ndarray | InitialPoint | None = None,
        initial_state: QuantumCircuit | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            estimator: The ``BaseEstimator`` class to use for the internal
                :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`.
            ansatz: The ``UCC`` ansatz. Its attributes ``qubit_mapper``, ``num_particles``,
                ``num_spatial_orbitals``, and ``initial_point`` will be completed at runtime based on
                the problem being solved.
            optimizer: The ``Optimizer`` or ``Minimizer`` to use for the internal
                :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`.
            initial_point: An optional initial point (i.e., initial parameter values for the VQE
                optimizer). If ``None`` then VQE will use an all-zero initial point of the
                appropriate length computed using
                :class:`~qiskit_nature.second_q.algorithms.initial_points.\
                hf_initial_point.HFInitialPoint`.
                This then defaults to the Hartree-Fock (HF) state when the HF circuit is prepended
                to the ansatz circuit. If another ``InitialPoint`` instance, this is used to
                compute an initial point for the VQE ansatz parameters. If a user-provided NumPy
                array, this is used directly.
            initial_state: Allows specification of a custom ``QuantumCircuit`` to be used as the
                initial state of the ansatz. If this is never set by the user, the factory will
                default to the :class:`~.HartreeFock` state.
            kwargs: Remaining keyword arguments are passed to the
                :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`.
        """
        warn_deprecated(
            "0.6.0",
            DeprecatedType.CLASS,
            "VQEUCCFactory",
            additional_msg=(
                ". This class is deprecated without replacement. Instead, refer to this how-to "
                "guide which explains the steps you need to take to replace the use of this class: "
                "https://qiskit.org/documentation/nature/howtos/vqe_ucc.html"
            ),
        )
        self._initial_state = initial_state
        self._initial_point = initial_point if initial_point is not None else HFInitialPoint()

        self._vqe = VQE(estimator, ansatz, optimizer, **kwargs)

    @property
    def ansatz(self) -> UCC:
        """Gets the user provided ansatz of future VQEs produced by the factory."""
        return self.minimum_eigensolver.ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UCC) -> None:
        """Sets the ansatz of future VQEs produced by the factory."""
        self.minimum_eigensolver.ansatz = ansatz

    @property
    def initial_point(self) -> np.ndarray | InitialPoint | None:
        """Gets the initial point of future VQEs produced by the factory."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray | InitialPoint | None) -> None:
        """Sets the initial point of future VQEs produced by the factory."""
        self._initial_point = initial_point

    @property
    def initial_state(self) -> QuantumCircuit | None:
        """
        Getter of the initial state.
        If value is ``None`` it will default to using the :class:`~.HartreeFock`.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit | None) -> None:
        """
        Setter of the initial state.
        If ``None`` is passed, this factory will default to using the :class:`~.HartreeFock`.
        """
        self._initial_state = initial_state

    @deprecate_arguments(
        "0.6.0",
        {"qubit_converter": "qubit_mapper"},
        additional_msg=(
            ". Additionally, the QubitConverter type in the qubit_mapper argument is deprecated "
            "and support for it will be removed together with the qubit_converter argument."
        ),
    )
    def get_solver(
        self,
        problem: ElectronicStructureProblem,
        qubit_mapper: QubitConverter | QubitMapper,
        *,
        qubit_converter: QubitConverter | QubitMapper | None = None,
    ) -> MinimumEigensolver:
        # pylint: disable=unused-argument
        """Returns a VQE with a UCC wavefunction ansatz, based on ``qubit_mapper``.

        Args:
            problem: A class encoding a problem to be solved.
            qubit_mapper: A class that converts second quantized operator to qubit operator.
                Providing a ``QubitConverter`` instance here is deprecated.
            qubit_converter: DEPRECATED A class that converts second quantized operator to qubit
                operator according to a mapper it is initialized with.

        Returns:
            A VQE suitable to compute the ground state of the molecule.
        """
        driver_result = problem
        num_spatial_orbitals = problem.num_spatial_orbitals
        num_particles = problem.num_alpha, problem.num_beta

        initial_state = self.initial_state
        if initial_state is None:
            initial_state = HartreeFock(num_spatial_orbitals, num_particles, qubit_mapper)

        self.ansatz.qubit_mapper = qubit_mapper
        self.ansatz.num_particles = num_particles
        self.ansatz.num_spatial_orbitals = num_spatial_orbitals
        self.ansatz.initial_state = initial_state

        if isinstance(self.initial_point, InitialPoint):
            self.initial_point.ansatz = self.ansatz
            self.initial_point.problem = driver_result
            initial_point = self.initial_point.to_numpy_array()
        else:
            initial_point = self.initial_point

        self.minimum_eigensolver.initial_point = initial_point
        return self.minimum_eigensolver

    def supports_aux_operators(self):
        return VQE.supports_aux_operators()

    @property
    def minimum_eigensolver(self) -> VQE:
        """Returns the solver instance."""
        return self._vqe
