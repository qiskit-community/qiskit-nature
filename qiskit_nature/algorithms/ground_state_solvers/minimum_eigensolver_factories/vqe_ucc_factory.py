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

import logging
import numpy as np

from qiskit.algorithms import MinimumEigensolver, VQE
from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import ExpectationBase
from qiskit.opflow.gradients import GradientBase
from qiskit.utils import QuantumInstance
from qiskit_nature.deprecation import deprecate_property, deprecate_method

from qiskit_nature.circuit.library import HartreeFock, UCC, UCCSD
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization.electronic import (
    ElectronicStructureProblem,
)
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber

from ...initial_points import InitialPoint
from .minimum_eigensolver_factory import MinimumEigensolverFactory

logger = logging.getLogger(__name__)


class VQEUCCFactory(MinimumEigensolverFactory):
    """A factory to construct a VQE minimum eigensolver with UCCSD ansatz wavefunction.

    Note: get_solver will overwrite any value that we directly set onto the vqe for both
    the ansatz and the initial point. For example:
    .. code-block:: python
        factory = VQEUCCFactory()
        factory.minimum_eigensolver.ansatz = UCCD()
        vqe = factory.get_solver()
        print(type(vqe.ansatz))  # UCCSD
    .. code-block:: python
    """

    def __init__(
        self,
        initial_point: Optional[Union[np.ndarray, InitialPoint]] = None,
        ansatz: Optional[UCC] = None,
        initial_state: Optional[QuantumCircuit] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            quantum_instance: The quantum instance used in the minimum eigensolver.
            optimizer: A classical optimizer.
            initial_point: An optional initial point (i.e., initial parameter values) for the
                optimizer. If ``None`` then VQE will use an all-zero initial point, which then
                defaults to the Hartree-Fock (HF) state when the HF circuit is prepended to the
                the ansatz circuit. If an
                :class:`~qiskit_nature.algorithms.initial_points.initial_point.InitialPoint`
                instance, this is used to compute an initial point for the VQE ansatz parameters.
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
        self._initial_state = initial_state
        self._initial_point = initial_point
        self._factory_ansatz = ansatz

        self._vqe = VQE(
            quantum_instance=kwargs.get("quantum_instance", None),
            optimizer=kwargs.get("optimizer", None),
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
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
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
        """DEPRECATED. Use ``minimum_eigensolver`` method and solver properties instead.
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
        """DEPRECATED. Use ``minimum_eigensolver`` and solver properties instead.
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

    @property
    def ansatz(self) -> Optional[UCC]:
        """Gets the ansatz of future VQEs produced by the factory."""
        return self._factory_ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[UCC]) -> None:
        """Sets the ansatz of future VQEs produced by the factory."""
        self._factory_ansatz = ansatz

    @property
    def initial_point(self) -> Optional[Union[np.ndarray, InitialPoint]]:
        """Gets the initial point of future VQEs produced by the factory."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[Union[np.ndarray, InitialPoint]]) -> None:
        """Sets the initial point of future VQEs produced by the factory."""
        self._initial_point = initial_point

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """Getter of the initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """Setter of the initial state. If ``None`` is passed, this factory will default to using
        the :class:`~.HartreeFock`."""
        self._initial_state = initial_state

    @property  # type: ignore
    @deprecate_method(
        "0.4", additional_msg="Use `minimum_eigensolver` and 'solver properties' instead."
    )
    def callback(self) -> Optional[Callable[[int, np.ndarray, float, float], None]]:
        """DEPRECATED. Use ``minimum_eigensolver`` and solver properties instead.
        Returns the callback."""
        return self.minimum_eigensolver.callback

    @callback.setter  # type: ignore
    @deprecate_property("0.4", additional_msg="Use the constructor instead.")
    def callback(self, callback: Optional[Callable[[int, np.ndarray, float, float], None]]) -> None:
        """DEPRECATED. Use the constructor instead.
        Sets the callback."""
        self.minimum_eigensolver.callback = callback

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

        ansatz = self.ansatz
        if ansatz is None:
            ansatz = UCCSD()
        ansatz.qubit_converter = qubit_converter
        ansatz.num_particles = num_particles
        ansatz.num_spin_orbitals = num_spin_orbitals
        ansatz.initial_state = initial_state
        self._vqe.ansatz = ansatz

        if isinstance(self.initial_point, InitialPoint):
            self.initial_point.grouped_property = driver_result
            self.initial_point.ansatz = ansatz
            # Override the initial_point with the computed array.
            self.initial_point = self.initial_point.to_numpy_array()

        self._vqe.initial_point = self.initial_point
        return self.minimum_eigensolver

    def supports_aux_operators(self):
        return VQE.supports_aux_operators()

    @property
    def minimum_eigensolver(self) -> VQE:
        """Returns the solver instance."""
        return self._vqe
