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
from qiskit_nature.second_q.circuit.library import UVCC, VSCF
from qiskit_nature.second_q.mappers import QubitConverter, QubitMapper
from qiskit_nature.second_q.problems import (
    VibrationalStructureProblem,
)
from qiskit_nature.deprecation import deprecate_arguments

from .minimum_eigensolver_factory import MinimumEigensolverFactory
from ...initial_points import InitialPoint, VSCFInitialPoint

logger = logging.getLogger(__name__)


class VQEUVCCFactory(MinimumEigensolverFactory):
    """DEPRECATED Factory to construct a :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`
    minimum eigensolver with :class:`~.UVCC` ansatz wavefunction.

    .. warning::

        This class is deprecated! Please see :ref:`this guide <how-to-vqe-uvcc>` for how to replace
        your usage of it!
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: UVCC,
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
            ansatz: The ``UVCC`` ansatz. Its attributes ``qubit_mapper``, ``num_modals``, and
                ``initial_point`` will be completed at runtime based on the problem being solved.
            optimizer: The ``Optimizer`` or ``Minimizer`` to use for the internal
                :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`.
            initial_point: An optional initial point (i.e., initial parameter values for the VQE
                optimizer). If ``None`` then VQE will use an all-zero initial point of the
                appropriate length computed using
                :class:`~initial_points.vscf_initial_point.VSCFInitialPoint`.
                This then defaults to the VSCF state when the VSCF circuit is prepended
                to the ansatz circuit. If another ``InitialPoint`` instance, this is used to
                compute an initial point for the VQE ansatz parameters. If a user-provided NumPy
                array, this is used directly.
            initial_state: Allows specification of a custom ``QuantumCircuit`` to be used as the
                initial state of the ansatz. If this is never set by the user, the factory will
                default to the :class:`~.VSCF` state.
            kwargs: Remaining keyword arguments are passed to the
                :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`.
        """
        warn_deprecated(
            "0.6.0",
            DeprecatedType.CLASS,
            "VQEUVCCFactory",
            additional_msg=(
                ". This class is deprecated without replacement. Instead, refer to this how-to "
                "guide which explains the steps you need to take to replace the use of this class: "
                "https://qiskit.org/documentation/nature/howtos/vqe_uvcc.html"
            ),
        )
        self._initial_state = initial_state
        self._initial_point = initial_point if initial_point is not None else VSCFInitialPoint()

        self._vqe = VQE(estimator, ansatz, optimizer, **kwargs)

    @property
    def ansatz(self) -> UVCC:
        """Gets the user provided ansatz of future VQEs produced by the factory."""
        return self.minimum_eigensolver.ansatz

    @ansatz.setter
    def ansatz(self, ansatz: UVCC) -> None:
        """Sets the ansatz of future VQEs produced by the factory."""
        self.minimum_eigensolver.ansatz = ansatz

    @property
    def initial_state(self) -> QuantumCircuit | None:
        """Getter of the initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit | None) -> None:
        """
        Setter of the initial state.
        If ``None`` is passed, this factory will default to using the :class:`~.VSCF`.
        """
        self._initial_state = initial_state

    @property
    def initial_point(self) -> np.ndarray | InitialPoint | None:
        """
        Gets the initial point of future VQEs produced by the factory.
        """
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray | InitialPoint | None) -> None:
        """Sets the initial point of future VQEs produced by the factory."""
        self._initial_point = initial_point

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
        problem: VibrationalStructureProblem,
        qubit_mapper: QubitConverter | QubitMapper,
        *,
        qubit_converter: QubitConverter | QubitMapper | None = None,
    ) -> MinimumEigensolver:
        # pylint: disable=unused-argument
        """Returns a VQE with a :class:`~.UVCC` wavefunction ansatz, based on ``qubit_mapper``.

        Args:
            problem: A class encoding a problem to be solved.
            qubit_mapper: A class that converts second quantized operator to qubit operator.
                Providing a ``QubitConverter`` instance here is deprecated.
            qubit_converter: DEPRECATED A class that converts second quantized operator to qubit
                operator according to a mapper it is initialized with.

        Returns:
            A VQE suitable to compute the ground state of the molecule.
        """
        initial_state = self.initial_state
        if initial_state is None:
            initial_state = VSCF(problem.num_modals)

        self.ansatz.qubit_mapper = qubit_mapper
        self.ansatz.num_modals = problem.num_modals
        self.ansatz.initial_state = initial_state

        if isinstance(self.initial_point, InitialPoint):
            self.initial_point.ansatz = self.ansatz
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
