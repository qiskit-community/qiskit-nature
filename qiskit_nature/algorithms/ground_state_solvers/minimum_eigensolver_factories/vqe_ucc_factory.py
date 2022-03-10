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

from typing import Optional, Union, Callable, List, Tuple, cast
import logging

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
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber, ElectronicEnergy
from qiskit_nature.initializers import MP2Initializer
from qiskit_nature.properties.second_quantization.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)

from .minimum_eigensolver_factory import MinimumEigensolverFactory

logger = logging.getLogger(__name__)


class VQEUCCFactory(MinimumEigensolverFactory):
    """A factory to construct a VQE minimum eigensolver with UCCSD ansatz wavefunction."""

    def __init__(
        self,
        quantum_instance: QuantumInstance,
        optimizer: Optional[Optimizer] = None,
        initial_point: Optional[Union[np.ndarray, str]] = None,
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
            initial_point: An optional initial point (i.e., initial parameter values)
                for the optimizer. If ``None`` then VQE will use an all-zero
                initial point, which then defaults to the Hartree-Fock (HF) state when
                the HF circuit is prepended to the beginning of the Ansatz circuit.
                If `"MP2"` then Moller-Plesset coefficients will be used for the double
                excitation coefficients of the initial point.
                See :class:`~qiskit_nature.initializers.MP2Initializer` for more info.
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
        self._quantum_instance = quantum_instance
        self._optimizer = optimizer
        self._initial_point = initial_point
        self._gradient = gradient
        self._expectation = expectation
        self._include_custom = include_custom
        self._ansatz = ansatz
        self._initial_state = initial_state
        self._callback = callback
        self._kwargs = kwargs

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Getter of the quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, q_instance: QuantumInstance) -> None:
        """Setter of the quantum instance."""
        self._quantum_instance = q_instance

    @property
    def optimizer(self) -> Optional[Optimizer]:
        """Getter of the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optional[Optimizer]) -> None:
        """Setter of the optimizer."""
        self._optimizer = optimizer

    @property
    def initializer(self) -> Optional[str]:
        """Getter of the initializer."""
        return self._initializer

    @initializer.setter
    def initializer(self, initializer: Optional[str]) -> None:
        """Setter of the initializer."""
        self._initializer = initializer

    @property
    def initial_point(self) -> Optional[Union[np.ndarray, str]]:
        """Getter of the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[Union[np.ndarray, str]]) -> None:
        """Setter of the initial point."""
        self._initial_point = initial_point

    @property
    def gradient(self) -> Optional[Union[GradientBase, Callable]]:
        """Getter of the gradient function"""
        return self._gradient

    @gradient.setter
    def gradient(self, gradient: Optional[Union[GradientBase, Callable]]) -> None:
        """Setter of the gradient function"""
        self._gradient = gradient

    @property
    def expectation(self) -> Optional[ExpectationBase]:
        """Getter of the expectation."""
        return self._expectation

    @expectation.setter
    def expectation(self, expectation: Optional[ExpectationBase]) -> None:
        """Setter of the expectation."""
        self._expectation = expectation

    @property
    def include_custom(self) -> bool:
        """Getter of the ``include_custom`` setting for the ``expectation`` setting."""
        return self._include_custom

    @include_custom.setter
    def include_custom(self, include_custom: bool) -> None:
        """Setter of the ``include_custom`` setting for the ``expectation`` setting."""
        self._include_custom = include_custom

    @property
    def ansatz(self) -> Optional[UCC]:
        """Getter of the ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[UCC]) -> None:
        """Setter of the ansatz. If ``None`` is passed, this factory will default to using the
        :class:`~.UCCSD` Ansatz."""
        self._ansatz = ansatz

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """Getter of the initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """Setter of the initial state. If ``None`` is passed, this factory will default to using
        the :class:`~.HartreeFock`."""
        self._initial_state = initial_state

    @property
    def callback(self) -> Optional[Callable[[int, np.ndarray, float, float], None]]:
        """Returns the callback."""
        return self._callback

    @callback.setter
    def callback(self, callback: Optional[Callable[[int, np.ndarray, float, float], None]]) -> None:
        """Sets the callback."""
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
                             according to a mapper it is initialized with

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

        initial_point = self._initial_point
        if not isinstance(initial_point, np.ndarray):
            # If a custom initial point is provided, keep it.
            # UCC ansatz must be built earlier to compute excitation list.
            ansatz._build()
            excitations = ansatz.excitation_list
            if initial_point is not None and initial_point.lower() == "mp2":
                initial_point = _get_mp2_initial_point(driver_result, excitations)
            else:
                initial_point = np.zeros(len(excitations))

        # Override initial point from args with computed value
        self._initial_point = initial_point

        # TODO: leverage re-usability of VQE after fixing
        # https://github.com/Qiskit/qiskit-terra/issues/7093
        vqe = VQE(
            ansatz=ansatz,
            quantum_instance=self.quantum_instance,
            optimizer=self.optimizer,
            initial_point=self.initial_point,
            gradient=self.gradient,
            expectation=self.expectation,
            include_custom=self.include_custom,
            callback=self.callback,
            **self._kwargs,
        )

        return vqe

    def supports_aux_operators(self):
        return VQE.supports_aux_operators()


def _get_mp2_initial_point(
    driver_result: GroupedSecondQuantizedProperty,
    excitations: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
) -> np.ndarray:
    """Get the intial point using MP2 double excitation coefficients.
    Returns all an all-zero array of the appropriate length if it cannot be computed.

    Args:
        driver_result: the second quantization properties from the driver.
        excitations: the list of excitations

    Returns:
        The initial point using MP2 double excitation coefficients.
    """
    electronic_energy = cast(ElectronicEnergy, driver_result.get_property(ElectronicEnergy))
    if electronic_energy is None:
        logger.warning("No ElectronicEnergy in driver result. Setting initial_point to all zeroes.")
        return np.zeros(len(excitations))

    particle_number = cast(ParticleNumber, driver_result.get_property(ParticleNumber))
    num_spin_orbitals = particle_number.num_spin_orbitals
    mp2_initializer = MP2Initializer(num_spin_orbitals, electronic_energy, excitations)
    return mp2_initializer.initial_point
