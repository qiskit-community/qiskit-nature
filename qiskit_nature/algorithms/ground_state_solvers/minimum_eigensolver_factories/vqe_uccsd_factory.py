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

from typing import Optional, Union, Callable

import numpy as np
from qiskit.utils import QuantumInstance
from qiskit.algorithms import MinimumEigensolver, VQE
from qiskit.opflow import ExpectationBase
from qiskit.opflow.gradients import GradientBase
from qiskit.algorithms.optimizers import Optimizer

from qiskit_nature.circuit.library.ansatzes import UCCSD
from qiskit_nature.circuit.library.initial_states import HartreeFock
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from ....problems.second_quantization.molecular.molecular_problem import MolecularProblem
from .minimum_eigensolver_factory import MinimumEigensolverFactory


class VQEUCCSDFactory(MinimumEigensolverFactory):
    """A factory to construct a VQE minimum eigensolver with UCCSD ansatz wavefunction."""

    def __init__(self,
                 quantum_instance: QuantumInstance,
                 optimizer: Optional[Optimizer] = None,
                 initial_point: Optional[np.ndarray] = None,
                 gradient: Optional[Union[GradientBase, Callable]] = None,
                 expectation: Optional[ExpectationBase] = None,
                 include_custom: bool = False,
                 method_singles: str = 'both',
                 method_doubles: str = 'ucc',
                 excitation_type: str = 'sd',
                 same_spin_doubles: bool = True) -> None:
        """
        Args:
            quantum_instance: The quantum instance used in the minimum eigensolver.
            optimizer: A classical optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the variational form for a
                preferred point and if not will simply compute a random one.
            gradient: An optional gradient function or operator for optimizer.
            expectation: The Expectation converter for taking the average value of the
                Observable over the var_form state function. When ``None`` (the default) an
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
            method_singles: specify the single excitation considered. 'alpha', 'beta',
                                'both' only alpha or beta spin-orbital single excitations or
                                both (all of them).
            method_doubles: specify the single excitation considered. 'ucc' (conventional
                                ucc), succ (singlet ucc), succ_full (singlet ucc full),
                                pucc (pair ucc).
            excitation_type: specify the excitation type 'sd', 's', 'd' respectively
                                for single and double, only single, only double excitations.
            same_spin_doubles: enable double excitations of the same spin.
        """
        self._quantum_instance = quantum_instance
        self._optimizer = optimizer
        self._initial_point = initial_point
        self._gradient = gradient
        self._expectation = expectation
        self._include_custom = include_custom
        self._method_singles = method_singles
        self._method_doubles = method_doubles
        self._excitation_type = excitation_type
        self._same_spin_doubles = same_spin_doubles
        self._vqe = VQE(var_form=None,
                        quantum_instance=self._quantum_instance,
                        optimizer=self._optimizer,
                        initial_point=self._initial_point,
                        gradient=self._gradient,
                        expectation=self._expectation,
                        include_custom=self._include_custom)

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
    def initial_point(self) -> Optional[np.ndarray]:
        """Getter of the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[np.ndarray]) -> None:
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
    def method_singles(self) -> str:
        """Getter of the ``method_singles`` setting for the ``method_singles`` setting."""
        return self._method_singles

    @method_singles.setter
    def method_singles(self, method_singles: str) -> None:
        """Setter of the ``method_singles`` setting for the ``method_singles`` setting."""
        self._method_singles = method_singles

    @property
    def method_doubles(self) -> str:
        """Getter of the ``method_doubles`` setting for the ``method_doubles`` setting."""
        return self._method_doubles

    @method_doubles.setter
    def method_doubles(self, method_doubles: str) -> None:
        """Setter of the ``method_doubles`` setting for the ``method_doubles`` setting."""
        self._method_doubles = method_doubles

    @property
    def excitation_type(self) -> str:
        """Getter of the ``excitation_type`` setting for the ``excitation_type`` setting."""
        return self._excitation_type

    @excitation_type.setter
    def excitation_type(self, excitation_type: str) -> None:
        """Setter of the ``excitation_type`` setting for the ``excitation_type`` setting."""
        self._excitation_type = excitation_type

    @property
    def same_spin_doubles(self) -> bool:
        """Getter of the ``same_spin_doubles`` setting for the ``same_spin_doubles`` setting."""
        return self._same_spin_doubles

    @same_spin_doubles.setter
    def same_spin_doubles(self, same_spin_doubles: bool) -> None:
        """Setter of the ``same_spin_doubles`` setting for the ``same_spin_doubles`` setting."""
        self._same_spin_doubles = same_spin_doubles

    def get_solver(self, problem: MolecularProblem,
                   qubit_converter: QubitConverter) -> MinimumEigensolver:
        """Returns a VQE with a UCCSD wavefunction ansatz, based on ``transformation``.
        This works only with a ``FermionicTransformation``.

        Args:
            problem: a class encoding a problem to be solved.
            qubit_converter: a class that converts second quantized operator to qubit operator
                             according to a mapper it is initialized with.

        Returns:
            A VQE suitable to compute the ground state of the molecule transformed
            by ``transformation``.

        Raises:
            QiskitNatureError: in case a Transformation of wrong type is given.
        """

        q_molecule_transformed = problem.q_molecule_transformed
        num_molecular_orbitals = q_molecule_transformed.num_molecular_orbitals
        num_particles = (q_molecule_transformed.num_alpha, q_molecule_transformed.num_beta)
        num_spin_orbitals = 2 * num_molecular_orbitals

        initial_state = HartreeFock(num_spin_orbitals, num_particles, qubit_converter)
        var_form = UCCSD(qubit_converter=qubit_converter,
                         num_particles=num_particles,
                         num_spin_orbitals=num_spin_orbitals,
                         initial_state=initial_state)
        self._vqe.var_form = var_form

        return self._vqe

    def supports_aux_operators(self):
        return VQE.supports_aux_operators()
