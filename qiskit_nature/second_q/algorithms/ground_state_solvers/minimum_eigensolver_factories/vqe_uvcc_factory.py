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

"""The minimum eigensolver factory for ground state calculation algorithms."""

import logging
from typing import Optional, Union, Callable, cast
import numpy as np

from qiskit.algorithms import MinimumEigensolver, VQE
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import ExpectationBase
from qiskit.opflow.gradients import GradientBase
from qiskit.utils import QuantumInstance
from qiskit_nature.circuit.library import UVCC, UVCCSD, VSCF
from qiskit_nature.second_q.operators import QubitConverter
from qiskit_nature.second_q.problems.vibrational import (
    VibrationalStructureProblem,
)
from qiskit_nature.second_q.operator_factories.vibrational import (
    VibrationalStructureDriverResult,
)
from qiskit_nature.deprecation import deprecate_property, deprecate_positional_arguments

from .minimum_eigensolver_factory import MinimumEigensolverFactory
from ...initial_points import InitialPoint, VSCFInitialPoint

logger = logging.getLogger(__name__)


class VQEUVCCFactory(MinimumEigensolverFactory):
    """Factory to construct a :class:`VQE` minimum eigensolver with :class:`UVCCSD` ansatz
    wavefunction.

    .. note::

       Any ansatz a user might directly set into VQE via the :attr:`minimum_eigensolver` will
       be overwritten by the factory when producing a solver via :meth:`get_solver`. This is
       due to the fact that the factory is designed to manage the ansatz and set it up according
       to the problem. Always pass any custom ansatz to be used when constructing the factory or
       by using its :attr:`ansatz` setter. The following code sample illustrates this behavior:

    .. code-block:: python

        from qiskit_nature.second_q.algorithms import VQEUVCCFactory
        from qiskit_nature.circuit.library import UVCCSD, UVCC
        factory = VQEUVCCFactory()
        vqe1 = factory.get_solver(problem, qubit_converter)
        print(type(vqe1.ansatz))  # UVCC()
        # Here the minimum_eigensolver ansatz just gets overwritten
        factory.minimum_eigensolver.ansatz = UVCC()
        vqe2 = factory.get_solver(problem, qubit_converter)
        print(type(vqe2.ansatz))  # UVCCSD
        # Here we change the factory ansatz and thus new VQEs are created with the new ansatz
        factory.ansatz = UVCC()
        vqe3 = factory.get_solver(problem, qubit_converter)
        print(type(vqe3.ansatz))  # UVCC

    """

    @deprecate_positional_arguments(
        version="0.4",
        func_name="VQEUVCCFactory Constructor",
        old_function_arguments=[
            "self",
            "quantum_instance",
            "optimizer",
            "initial_point",
            "gradient",
            "expectation",
            "include_custom",
            "ansatz",
            "initial_state",
            "callback",
        ],
        stack_level=5,
    )
    def __init__(
        self,
        initial_point: Optional[Union[np.ndarray, InitialPoint]] = None,
        ansatz: Optional[UVCC] = None,
        initial_state: Optional[QuantumCircuit] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            initial_point: An optional initial point (i.e., initial parameter values for the VQE
                optimizer). If ``None`` then VQE will use an all-zero initial point of the
                appropriate length computed using
                :class:`~qiskit_nature.second_q.algorithms.initial_points.vscf_initial_point.VSCFInitialPoint`.
                This then defaults to the VSCF state when the VSCF circuit is prepended
                to the the ansatz circuit. If another
                :class:`~qiskit_nature.second_q.algorithms.initial_points.initial_point.InitialPoint`
                instance, this is used to compute an initial point for the VQE ansatz parameters.
                If a user-provided NumPy array, this is used directly.
            initial_state: Allows specification of a custom `QuantumCircuit` to be used as the
                initial state of the ansatz. If this is never set by the user, the factory will
                default to the :class:`~.VSCF` state.
            ansatz: Allows specification of a custom :class:`~.UCC` instance. This defaults to None
                where the factory will internally create and use a :class:`~.UVCCSD` ansatz.
            kwargs: Remaining keyword arguments are passed to the :class:`VQE`.
        """

        self._initial_state = initial_state
        self._initial_point = initial_point if initial_point is not None else VSCFInitialPoint()
        self._ansatz = ansatz

        self._vqe = VQE(**kwargs)

    @property  # type: ignore
    @deprecate_property("0.4", additional_msg="Use `minimum_eigensolver.quantum_instance` instead.")
    def quantum_instance(self) -> QuantumInstance:
        """DEPRECATED. Use ``minimum_eigensolver.quantum_instance`` instead.
        Returns quantum instance."""
        return self.minimum_eigensolver.quantum_instance

    @quantum_instance.setter  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver.quantum_instance = q_instance` instead."
    )
    def quantum_instance(self, q_instance: QuantumInstance) -> None:
        """DEPRECATED. Use `minimum_eigensolver.quantum_instance` instead. Sets the quantum instance."""
        self.minimum_eigensolver.quantum_instance = q_instance

    @property  # type: ignore
    @deprecate_property("0.4", additional_msg="Use `minimum_eigensolver.optimizer` instead.")
    def optimizer(self) -> Optional[Optimizer]:
        """DEPRECATED. Use ``minimum_eigensolver.optimizer``  instead.
        Returns optimizer."""
        return self.minimum_eigensolver.optimizer

    @optimizer.setter  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver.optimizer = optimizer` instead."
    )
    def optimizer(self, optimizer: Optional[Optimizer]) -> None:
        """DEPRECATED. Use `minimum_eigensolver.optimizer = optimizer` instead. Sets the optimizer."""
        self.minimum_eigensolver.optimizer = optimizer

    @property  # type: ignore
    @deprecate_property("0.4", additional_msg="Use `minimum_eigensolver.gradient` instead.")
    def gradient(self) -> Optional[Union[GradientBase, Callable]]:
        """DEPRECATED. Use ``minimum_eigensolver.gradient`` instead.
        Returns gradient."""
        return self.minimum_eigensolver.gradient

    @gradient.setter  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver.gradient = gradient` instead."
    )
    def gradient(self, gradient: Optional[Union[GradientBase, Callable]]) -> None:
        """DEPRECATED. Use `minimum_eigensolver.gradient = gradient` instead. Sets the gradient."""
        self.minimum_eigensolver.gradient = gradient

    @property  # type: ignore
    @deprecate_property("0.4", additional_msg="Use `minimum_eigensolver.expectation` instead.")
    def expectation(self) -> Optional[ExpectationBase]:
        """DEPRECATED. Use ``minimum_eigensolver.expectation`` instead.
        Returns expectation."""
        return self.minimum_eigensolver.expectation

    @expectation.setter  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver.expectation = expectation` instead."
    )
    def expectation(self, expectation: Optional[ExpectationBase]) -> None:
        """DEPRECATED. Use `minimum_eigensolver.expectation = expectation` instead. Sets expectation."""
        self.minimum_eigensolver.expectation = expectation

    @property  # type: ignore
    @deprecate_property("0.4", additional_msg="Use `minimum_eigensolver.include_custom` instead.")
    def include_custom(self) -> bool:
        """DEPRECATED. Use ``minimum_eigensolver.include_custom`` instead.
        Getter of the ``include_custom`` setting for the ``expectation`` setting."""
        return self.minimum_eigensolver.include_custom

    @include_custom.setter  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver.include_custom = include_custom` instead."
    )
    def include_custom(self, include_custom: bool) -> None:
        """DEPRECATED. Use `minimum_eigensolver.include_custom = include_custom` instead.
        Setter of the ``include_custom``
        setting for the ``expectation`` setting."""
        self.minimum_eigensolver.include_custom = include_custom

    @property
    def ansatz(self) -> Optional[UVCC]:
        """
        Gets the user provided ansatz of future VQEs produced by the factory.
        If value is ``None`` it defaults to :class:`~.UVCCSD`.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[UVCC]) -> None:
        """
        Sets the ansatz of future VQEs produced by the factory.
        If set to ``None`` it defaults to :class:`~.UVCCSD`.
        """
        self._ansatz = ansatz

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """Getter of the initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """
        Setter of the initial state.
        If ``None`` is passed, this factory will default to using the :class:`~.VSCF`.
        """
        self._initial_state = initial_state

    @property
    def initial_point(self) -> Optional[Union[np.ndarray, InitialPoint]]:
        """
        Gets the initial point of future VQEs produced by the factory.
        """
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[Union[np.ndarray, InitialPoint]]) -> None:
        """Sets the initial point of future VQEs produced by the factory."""
        self._initial_point = initial_point

    @property  # type: ignore
    @deprecate_property("0.4", additional_msg="Use `minimum_eigensolver.callback` instead.")
    def callback(self) -> Optional[Callable[[int, np.ndarray, float, float], None]]:
        """Returns the callback."""
        return self.minimum_eigensolver.callback

    @callback.setter  # type: ignore
    @deprecate_property("0.4", additional_msg="Use the constructor instead.")
    def callback(self, callback: Optional[Callable[[int, np.ndarray, float, float], None]]) -> None:
        """DEPRECATED. Use the constructor instead.
        Sets the callback."""
        self.minimum_eigensolver.callback = callback

    def get_solver(  # type: ignore[override]
        self,
        problem: VibrationalStructureProblem,
        qubit_converter: QubitConverter,
    ) -> MinimumEigensolver:
        """Returns a VQE with a :class:`~.UVCCSD` wavefunction ansatz, based on ``qubit_converter``.

        Args:
            problem: a class encoding a problem to be solved.
            qubit_converter: a class that converts second quantized operator to qubit operator
                             according to a mapper it is initialized with.

        Returns:
            A VQE suitable to compute the ground state of the molecule.
        """

        basis = cast(VibrationalStructureDriverResult, problem.grouped_property_transformed).basis
        num_modals = basis.num_modals_per_mode
        num_modes = len(num_modals)

        if isinstance(num_modals, int):
            num_modals = [num_modals] * num_modes

        initial_state = self.initial_state
        if initial_state is None:
            initial_state = VSCF(num_modals)

        ansatz = self._ansatz
        if ansatz is None:
            ansatz = UVCCSD()
        ansatz.qubit_converter = qubit_converter
        ansatz.num_modals = num_modals
        ansatz.initial_state = initial_state

        if isinstance(self.initial_point, InitialPoint):
            self.initial_point.ansatz = ansatz
            initial_point = self.initial_point.to_numpy_array()
        else:
            initial_point = self.initial_point

        self.minimum_eigensolver.initial_point = initial_point
        self.minimum_eigensolver.ansatz = ansatz
        return self.minimum_eigensolver

    def supports_aux_operators(self):
        return VQE.supports_aux_operators()

    @property
    def minimum_eigensolver(self) -> VQE:
        """Returns the solver instance."""
        return self._vqe
