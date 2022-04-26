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


from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization.vibrational import (
    VibrationalStructureProblem,
)
from qiskit_nature.properties.second_quantization.vibrational import (
    VibrationalStructureDriverResult,
)
from qiskit_nature.deprecation import deprecate_property

from .minimum_eigensolver_factory import MinimumEigensolverFactory

logger = logging.getLogger(__name__)


class VQEUVCCFactory(MinimumEigensolverFactory):
    """A factory to construct a VQE minimum eigensolver with UVCCSD ansatz wavefunction."""

    def __init__(
        self,
        ansatz: Optional[UVCC] = None,
        initial_state: Optional[QuantumCircuit] = None,
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
        self._initial_state = initial_state

        self._vqe = VQE(
            ansatz=ansatz,
            quantum_instance=kwargs.get("quantum_instance"),
            optimizer=kwargs.get("optimizer", None),
            initial_point=kwargs.get("initial_point", None),
            gradient=kwargs.get("gradient", None),
            expectation=kwargs.get("expectation", None),
            include_custom=kwargs.get("include_custom", False),
            callback=kwargs.get("callback", None),
        )

    @property  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver` and 'solver properties' instead."
    )
    def quantum_instance(self) -> QuantumInstance:
        """DEPRECATED. Use ``minimum_eigensolver`` and solver properties instead.
        Returns quantum instance."""
        return self.minimum_eigensolver.quantum_instance

    @quantum_instance.setter  # type: ignore
    @deprecate_property("0.4", additional_msg="Use the constructor instead.")
    def quantum_instance(self, q_instance: QuantumInstance) -> None:
        """DEPRECATED. Use the constructor instead. Sets the quantum instance."""
        self.minimum_eigensolver.quantum_instance = q_instance

    @property  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver` and 'solver properties' instead."
    )
    def optimizer(self) -> Optional[Optimizer]:
        """DEPRECATED. Use ``minimum_eigensolver`` and solver properties instead.
        Returns optimizer."""
        return self.minimum_eigensolver.optimizer

    @optimizer.setter  # type: ignore
    @deprecate_property("0.4", additional_msg="Use the constructor instead.")
    def optimizer(self, optimizer: Optional[Optimizer]) -> None:
        """DEPRECATED. Use the constructor instead. Sets the optimizer."""
        self.minimum_eigensolver.optimizer = optimizer

    @property  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver` and 'solver properties' instead."
    )
    def initial_point(self) -> Optional[np.ndarray]:
        """DEPRECATED. Use ``minimum_eigensolver`` and solver properties instead.
        Returns initial_point."""
        return self.minimum_eigensolver.initial_point

    @initial_point.setter  # type: ignore
    @deprecate_property("0.4", additional_msg="Use the constructor instead.")
    def initial_point(self, initial_point: Optional[np.ndarray]) -> None:
        """DEPRECATED. Use the constructor instead. Sets the initial_point."""
        self.minimum_eigensolver.initial_point = initial_point

    @property  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver` and 'solver properties' instead."
    )
    def gradient(self) -> Optional[Union[GradientBase, Callable]]:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Returns gradient."""
        return self.minimum_eigensolver.gradient

    @gradient.setter  # type: ignore
    @deprecate_property("0.4", additional_msg="Use the constructor instead.")
    def gradient(self, gradient: Optional[Union[GradientBase, Callable]]) -> None:
        """DEPRECATED. Use the constructor instead. Sets the initial_point."""
        self.minimum_eigensolver.gradient = gradient

    @property  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver` and 'solver properties' instead."
    )
    def expectation(self) -> Optional[ExpectationBase]:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Returns gradient."""
        return self.minimum_eigensolver.expectation

    @expectation.setter  # type: ignore
    @deprecate_property("0.4", additional_msg="Use the constructor instead.")
    def expectation(self, expectation: Optional[ExpectationBase]) -> None:
        """DEPRECATED. Use the constructor instead. Sets the initial_point."""
        self.minimum_eigensolver.expectation = expectation

    @property  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver` and 'solver properties' instead."
    )
    def include_custom(self) -> bool:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Getter of the ``include_custom`` setting for the ``expectation`` setting."""
        return self.minimum_eigensolver.include_custom

    @include_custom.setter  # type: ignore
    @deprecate_property("0.4", additional_msg="Use the constructor instead.")
    def include_custom(self, include_custom: bool) -> None:
        """DEPRECATED. Use the constructor instead. Setter of the ``include_custom``
        setting for the ``expectation`` setting."""
        self.minimum_eigensolver.include_custom = include_custom

    @property  # type: ignore
    @deprecate_property(
        "0.4", additional_msg="Use `minimum_eigensolver` and 'solver properties' instead."
    )
    def ansatz(self) -> Optional[UVCC]:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Getter of the ansatz"""
        return self.minimum_eigensolver.ansatz

    @ansatz.setter  # type: ignore
    @deprecate_property("0.4", additional_msg="Use the constructor instead.")
    def ansatz(self, ansatz: Optional[UVCC]) -> None:
        """DEPRECATED. Use the constructor instead. Setter of the ``include_custom``
        Setter of the ansatz. If ``None`` is passed, this factory will default to using the
        :class:`~.UCCSD` Ansatz."""
        self.minimum_eigensolver.ansatz = ansatz

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """Getter of the initial state."""
        return self._initial_state

    @initial_state.setter  # type: ignore
    @deprecate_property("0.4", additional_msg="Use the constructor instead.")
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """DEPRECATED. Use the constructor instead.
        Setter of the initial state. If ``None`` is passed, this factory will default to using
        the :class:`~.HartreeFock`."""
        self._initial_state = initial_state

    @property
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
        return self.minimum_eigensolver

    def supports_aux_operators(self):
        return VQE.supports_aux_operators()

    @property
    def minimum_eigensolver(self) -> VQE:
        """Returns the solver instance."""
        return self._vqe
