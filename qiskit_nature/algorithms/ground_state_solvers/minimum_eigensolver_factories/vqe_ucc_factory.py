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

from typing import Optional, Union, Callable, cast

import warnings
import numpy as np
from qiskit.algorithms import MinimumEigensolver, VQE
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import ExpectationBase
from qiskit.opflow.gradients import GradientBase
from qiskit.utils import QuantumInstance

from qiskit_nature.circuit.library import HartreeFock, UCC, UCCSD
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization.electronic import (
    ElectronicStructureProblem,
)
from qiskit_nature.properties.second_quantization.electronic import (
    ParticleNumber,
)
from .minimum_eigensolver_factory import MinimumEigensolverFactory


class VQEUCCFactory(MinimumEigensolverFactory):
    """A factory to construct a VQE minimum eigensolver with UCCSD ansatz wavefunction."""

    def __init__(
        self,
        quantum_instance: QuantumInstance,
        optimizer: Optional[Optimizer] = None,
        initial_point: Optional[np.ndarray] = None,
        gradient: Optional[Union[GradientBase, Callable]] = None,
        expectation: Optional[ExpectationBase] = None,
        include_custom: bool = False,
        ansatz: Optional[UCC] = None,
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
            ansatz: Allows specification of a custom :class:`~.UCC` instance. If this is never set
                by the user, the factory will default to the :class:`~.UCCSD` Ansatz.
            initial_state: Allows specification of a custom `QuantumCircuit` to be used as the
                initial state of the ansatz. If this is never set by the user, the factory will
                default to the :class:`~.HartreeFock` state.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.`
            kwargs: any additional keyword arguments will be passed on to the VQE.
        """
        self.ansatz = ansatz
        self.initial_state = initial_state

        # Not needed
        self.quantum_instance = quantum_instance
        self.optimizer = optimizer
        self.initial_point = initial_point
        self.gradient = gradient
        self.expectation = expectation
        self.include_custom = include_custom
        self.callback = callback
        self._kwargs = kwargs

        self._vqe = VQE(
            quantum_instance=quantum_instance,
            optimizer=optimizer,
            initial_point=initial_point,
            gradient=gradient,
            expectation=expectation,
            include_custom=include_custom,
            callback=callback,
            **kwargs,
        )


    @property
    def quantum_instance(self) -> QuantumInstance:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Returns quantum instance."""
        warnings.warn(
            "The `quantum_instance` getter method is deprecated as of Qiskit Nature 0.4, "
            "and will be removed in a future release. Use ``minimum_eigensolver`` method and "
            "solver properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, q_instance: QuantumInstance) -> None:
        """DEPRECATED. Use the constructor instead. Sets the quantum instance."""
        warnings.warn(
            "The `quantum_instance` setter method is deprecated as of Qiskit Nature 0.4 "
            "and will be removed in a future release. Use the constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._quantum_instance = q_instance

    @property
    def optimizer(self) -> Optional[Optimizer]:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Returns optimizer."""
        warnings.warn(
            "The `optimizer` getter method is deprecated as of Qiskit Nature 0.4, "
            "and will be removed in a future release. Use ``minimum_eigensolver`` method and "
            "solver properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optional[Optimizer]) -> None:
        """DEPRECATED. Use the constructor instead. Sets the optimizer."""
        warnings.warn(
            "The `optimizer` setter method is deprecated as of Qiskit Nature 0.4 "
            "and will be removed in a future release. Use the constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._optimizer = optimizer

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Returns initial_point."""
        warnings.warn(
            "The `initial_point` getter method is deprecated as of Qiskit Nature 0.4, "
            "and will be removed in a future release. Use ``minimum_eigensolver`` method and "
            "solver properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[np.ndarray]) -> None:
        """DEPRECATED. Use the constructor instead. Sets the initial_point."""
        warnings.warn(
            "The `initial_point` setter method is deprecated as of Qiskit Nature 0.4 "
            "and will be removed in a future release. Use the constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._initial_point = initial_point

    @property
    def gradient(self) -> Optional[Union[GradientBase, Callable]]:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Returns gradient."""
        warnings.warn(
            "The `gradient` getter method is deprecated as of Qiskit Nature 0.4, "
            "and will be removed in a future release. Use ``minimum_eigensolver`` method and "
            "solver properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._gradient

    @gradient.setter
    def gradient(self, gradient: Optional[Union[GradientBase, Callable]]) -> None:
        """DEPRECATED. Use the constructor instead. Sets the initial_point."""
        warnings.warn(
            "The `gradient` setter method is deprecated as of Qiskit Nature 0.4 "
            "and will be removed in a future release. Use the constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._gradient = gradient

    @property
    def expectation(self) -> Optional[ExpectationBase]:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Returns gradient."""
        warnings.warn(
            "The `expectation` getter method is deprecated as of Qiskit Nature 0.4, "
            "and will be removed in a future release. Use ``minimum_eigensolver`` method and "
            "solver properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._expectation

    @expectation.setter
    def expectation(self, expectation: Optional[ExpectationBase]) -> None:
        """DEPRECATED. Use the constructor instead. Sets the initial_point."""
        warnings.warn(
            "The `expectation` setter method is deprecated as of Qiskit Nature 0.4 "
            "and will be removed in a future release. Use the constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._expectation = expectation

    @property
    def include_custom(self) -> bool:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Getter of the ``include_custom`` setting for the ``expectation`` setting."""
        warnings.warn(
            "The `include_custom` getter method is deprecated as of Qiskit Nature 0.4, "
            "and will be removed in a future release. Use ``minimum_eigensolver`` method and "
            "solver properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._include_custom

    @include_custom.setter
    def include_custom(self, include_custom: bool) -> None:
        """DEPRECATED. Use the constructor instead. SSetter of the ``include_custom``
        setting for the ``expectation`` setting."""
        warnings.warn(
            "The `include_custom` setter method is deprecated as of Qiskit Nature 0.4 "
            "and will be removed in a future release. Use the constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._include_custom = include_custom

    @property
    def ansatz(self) -> Optional[UCC]:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Getter of the ansatz"""
        warnings.warn(
            "The `ansatz` getter method is deprecated as of Qiskit Nature 0.4, "
            "and will be removed in a future release. Use ``minimum_eigensolver`` method and "
            "solver properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[UCC]) -> None:
        """DEPRECATED. Use the constructor instead. SSetter of the ``include_custom``
        Setter of the ansatz. If ``None`` is passed, this factory will default to using the
        :class:`~.UCCSD` Ansatz."""
        warnings.warn(
            "The `include_custom` setter method is deprecated as of Qiskit Nature 0.4 "
            "and will be removed in a future release. Use the constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._ansatz = ansatz

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Getter of the initial state."""
        warnings.warn(
            "The `initial_state` getter method is deprecated as of Qiskit Nature 0.4, "
            "and will be removed in a future release. Use ``minimum_eigensolver`` method and "
            "solver properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """DEPRECATED. Use the constructor instead.
        Setter of the initial state. If ``None`` is passed, this factory will default to using
        the :class:`~.HartreeFock`."""
        warnings.warn(
            "The `include_custom` setter method is deprecated as of Qiskit Nature 0.4 "
            "and will be removed in a future release. Use the constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._initial_state = initial_state

    @property
    def callback(self) -> Optional[Callable[[int, np.ndarray, float, float], None]]:
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
        Returns the callback."""
        warnings.warn(
            "The `initial_state` getter method is deprecated as of Qiskit Nature 0.4, "
            "and will be removed in a future release. Use ``minimum_eigensolver`` method and "
            "solver properties instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._callback

    @callback.setter
    def callback(self, callback: Optional[Callable[[int, np.ndarray, float, float], None]]) -> None:
        """DEPRECATED. Use the constructor instead.
        Sets the callback."""
        warnings.warn(
            "The `include_custom` setter method is deprecated as of Qiskit Nature 0.4 "
            "and will be removed in a future release. Use the constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._callback = callback

    def get_solver(  # type: ignore[override]
        self,
        problem: ElectronicStructureProblem,
        qubit_converter: QubitConverter,
    ) -> MinimumEigensolver:
        """Returns a VQE with a UCCSD wavefunction ansatz, based on ``qubit_converter``.

        Args:
            problem: a class encoding a problem to be solved.
            qubit_converter: a class that converts second quantized operator to qubit operator
                             according to a mapper it is initialized with.

        Returns:
            A VQE suitable to compute the ground state of the molecule.
        """
        driver_result = problem.grouped_property_transformed
        particle_number = cast(ParticleNumber, driver_result.get_property(ParticleNumber))
        num_spin_orbitals = particle_number.num_spin_orbitals
        num_particles = particle_number.num_alpha, particle_number.num_beta

        initial_state = self.initial_state
        if initial_state is None:
            initial_state = HartreeFock(num_spin_orbitals, num_particles, qubit_converter)

        if self._vqe.ansatz is None:
            self._vqe.ansatz = UCCSD()
        self._vqe.ansatz.qubit_converter = qubit_converter
        self._vqe.ansatz.num_particles = num_particles
        self._vqe.ansatz.num_spin_orbitals = num_spin_orbitals
        self._vqe.ansatz.initial_state = initial_state

        return self._vqe

    def supports_aux_operators(self):
        return VQE.supports_aux_operators()

    @property
    def minimum_eigensolver(self):
        """Returns the solver instance."""
        return self._vqe