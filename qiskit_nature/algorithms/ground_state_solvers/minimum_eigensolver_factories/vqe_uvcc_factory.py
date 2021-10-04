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

"""The minimum eigensolver factory for ground state calculation algorithms."""

from typing import Optional, Union, Callable, cast
import numpy as np

from qiskit.algorithms import MinimumEigensolver, VQE
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import ExpectationBase
from qiskit.opflow.gradients import GradientBase
from qiskit.utils import QuantumInstance
from qiskit_nature.circuit.library import UVCC, UVCCSD, VSCF
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization.vibrational import (
    VibrationalStructureProblem,
)
from qiskit_nature.properties.second_quantization.vibrational import (
    VibrationalStructureDriverResult,
)

from .minimum_eigensolver_factory import MinimumEigensolverFactory


class VQEUVCCFactory(MinimumEigensolverFactory):
    """A factory to construct a VQE minimum eigensolver with UVCCSD ansatz wavefunction."""

    def __init__(
        self,
        quantum_instance: QuantumInstance,
        optimizer: Optional[Optimizer] = None,
        initial_point: Optional[np.ndarray] = None,
        gradient: Optional[Union[GradientBase, Callable]] = None,
        expectation: Optional[ExpectationBase] = None,
        include_custom: bool = False,
        ansatz: Optional[UVCC] = None,
        initial_state: Optional[QuantumCircuit] = None,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            quantum_instance: The quantum instance used in the minimum eigensolver.
            optimizer: A classical optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            gradient: An optional gradient function or operator for optimizer.
            expectation: The Expectation converter for taking the average value of the
                Observable over the ansatz state function. When ``None`` (the default) an
                :class:`~qiskit.opflow.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            include_custom: When `expectation` parameter here is None setting this to ``True`` will
                allow the factory to include the custom Aer pauli expectation.
            ansatz: Allows specification of a custom :class:`~.UVCC` instance. If this is never
                set by the user, the factory will default to the :class:`~.UVCCSD` Ansatz.
            initial_state: Allows specification of a custom `QuantumCircuit` to be used as the
                initial state of the ansatz. If this is never set by the user, the factory will
                default to the :class:`~.VSCF` state.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.`
            kwargs: any additional keyword arguments will be passed on to the VQE.
        """
        self._vqe = VQE(
            **kwargs,
        )
        self.quantum_instance = quantum_instance
        self.optimizer = optimizer
        self.initial_point = initial_point
        self.gradient = gradient
        self.expectation = expectation
        self.include_custom = include_custom
        self.ansatz = ansatz
        self.initial_state = initial_state
        self.callback = callback

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Getter of the quantum instance."""
        return self._vqe.quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, q_instance: QuantumInstance) -> None:
        """Setter of the quantum instance."""
        self._vqe.quantum_instance = q_instance

    @property
    def optimizer(self) -> Optional[Optimizer]:
        """Getter of the optimizer."""
        return self._vqe.optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optional[Optimizer]) -> None:
        """Setter of the optimizer."""
        self._vqe.optimizer = optimizer

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """Getter of the initial point."""
        return self._vqe.initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[np.ndarray]) -> None:
        """Setter of the initial point."""
        self._vqe.initial_point = initial_point

    @property
    def gradient(self) -> Optional[Union[GradientBase, Callable]]:
        """Getter of the gradient function"""
        return self._vqe.gradient

    @gradient.setter
    def gradient(self, gradient: Optional[Union[GradientBase, Callable]]) -> None:
        """Setter of the gradient function"""
        self._vqe.gradient = gradient

    @property
    def expectation(self) -> Optional[ExpectationBase]:
        """Getter of the expectation."""
        return self._vqe.expectation

    @expectation.setter
    def expectation(self, expectation: Optional[ExpectationBase]) -> None:
        """Setter of the expectation."""
        self._vqe.expectation = expectation

    # TODO: remove dependence on private VQE variable
    # https://github.com/Qiskit/qiskit-terra/issues/7093
    @property
    def include_custom(self) -> bool:
        """Getter of the ``include_custom`` setting for the ``expectation`` setting."""
        return self._vqe._include_custom

    @include_custom.setter
    def include_custom(self, include_custom: bool) -> None:
        """Setter of the ``include_custom`` setting for the ``expectation`` setting."""
        self._vqe._include_custom = include_custom

    @property
    def ansatz(self) -> Optional[UVCC]:
        """Getter of the ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[UVCC]) -> None:
        """Setter of the ansatz. If ``None`` is passed, this factory will default to using the
        :class:`~.UVCCSD` Ansatz."""
        self._ansatz = ansatz

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """Getter of the initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """Setter of the initial state. If ``None`` is passed, this factory will default to using
        the :class:`~.VSCF`."""
        self._initial_state = initial_state

    # TODO: remove dependence on private VQE variable
    # https://github.com/Qiskit/qiskit-terra/issues/7093
    @property
    def callback(self) -> Optional[Callable[[int, np.ndarray, float, float], None]]:
        """Returns the callback."""
        return self._vqe._callback

    @callback.setter
    def callback(self, callback: Optional[Callable[[int, np.ndarray, float, float], None]]) -> None:
        """Sets the callback."""
        self._vqe._callback = callback

    def get_solver(  # type: ignore[override]
        self,
        problem: VibrationalStructureProblem,
        qubit_converter: QubitConverter,
    ) -> MinimumEigensolver:
        """Returns a VQE with a UVCCSD wavefunction ansatz, based on ``qubit_converter``.

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

        ansatz = self.ansatz
        if ansatz is None:
            ansatz = UVCCSD()
        ansatz.qubit_converter = qubit_converter
        ansatz.num_modals = num_modals
        ansatz.initial_state = initial_state

        self._vqe.ansatz = ansatz

        return self._vqe

    def supports_aux_operators(self):
        return VQE.supports_aux_operators()
