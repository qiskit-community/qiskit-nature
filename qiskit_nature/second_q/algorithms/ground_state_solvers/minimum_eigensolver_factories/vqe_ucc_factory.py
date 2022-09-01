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

from typing import Optional, Union
import logging
import numpy as np

from qiskit.algorithms import MinimumEigensolver, VQE
from qiskit.circuit import QuantumCircuit

from qiskit_nature.second_q.circuit.library import HartreeFock, UCC, UCCSD
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.problems import (
    ElectronicStructureProblem,
)

from ...initial_points import InitialPoint, HFInitialPoint
from .minimum_eigensolver_factory import MinimumEigensolverFactory

logger = logging.getLogger(__name__)


class VQEUCCFactory(MinimumEigensolverFactory):
    """Factory to construct a :class:`~qiskit.algorithms.VQE` minimum eigensolver with :class:`~.UCCSD`
    ansatz wavefunction.

    .. note::

       Any ansatz a user might directly set into VQE via the :attr:`minimum_eigensolver` will
       be overwritten by the factory when producing a solver via :meth:`get_solver`. This is
       due to the fact that the factory is designed to manage the ansatz and set it up according
       to the problem. Always pass any custom ansatz to be used when constructing the factory or
       by using its :attr:`ansatz` setter. The following code sample illustrates this behavior:

    .. code-block:: python

        from qiskit_nature.second_q.algorithms import VQEUCCFactory
        from qiskit_nature.second_q.circuit.library import UCCSD, UCC
        factory = VQEUCCFactory()
        vqe1 = factory.get_solver(problem, qubit_converter)
        print(type(vqe1.ansatz))  # UCCSD (default)
        vqe1.ansatz = UCC()
        # Here the minimum_eigensolver ansatz just gets overwritten
        factory.minimum_eigensolver.ansatz = UCC()
        vqe2 = factory.get_solver(problem, qubit_converter)
        print(type(vqe2.ansatz))  # UCCSD
        # Here we change the factory ansatz and thus new VQEs are created with the new ansatz
        factory.ansatz = UCC()
        vqe3 = factory.get_solver(problem, qubit_converter)
        print(type(vqe3.ansatz))  # UCC

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
            initial_point: An optional initial point (i.e., initial parameter values for the VQE
                optimizer). If ``None`` then VQE will use an all-zero initial point of the
                appropriate length computed using
                :class:`~qiskit_nature.second_q.algorithms.initial_points.\
                hf_initial_point.HFInitialPoint`.
                This then defaults to the Hartree-Fock (HF) state when the HF circuit is prepended
                to the ansatz circuit. If another
                :class:`~qiskit_nature.second_q.algorithms.initial_points.initial_point.InitialPoint`
                instance, this is used to compute an initial point for the VQE ansatz parameters.
                If a user-provided NumPy array, this is used directly.
            initial_state: Allows specification of a custom `QuantumCircuit` to be used as the
                initial state of the ansatz. If this is never set by the user, the factory will
                default to the :class:`~.HartreeFock` state.
            ansatz: Allows specification of a custom :class:`~.UCC` instance. This defaults to None
                where the factory will internally create and use a :class:`~.UCCSD` ansatz.
            kwargs: Remaining keyword arguments are passed to the :class:`~.VQE`.
        """

        self._initial_state = initial_state
        self.initial_point = initial_point if initial_point is not None else HFInitialPoint()
        self._ansatz = ansatz

        self._vqe = VQE(**kwargs)

    @property
    def ansatz(self) -> Optional[UCC]:
        """
        Gets the user provided ansatz of future VQEs produced by the factory.
        If value is ``None`` it defaults to :class:`~.UCCSD`.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[UCC]) -> None:
        """
        Sets the ansatz of future VQEs produced by the factory.
        If set to ``None`` it defaults to :class:`~.UCCSD`.
        """
        self._ansatz = ansatz

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
        """
        Getter of the initial state.
        If value is ``None`` it will default to using the :class:`~.HartreeFock`.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """
        Setter of the initial state.
        If ``None`` is passed, this factory will default to using the :class:`~.HartreeFock`.
        """
        self._initial_state = initial_state

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
        driver_result = problem
        particle_number = driver_result.properties.particle_number
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
        self.minimum_eigensolver.ansatz = ansatz

        if isinstance(self.initial_point, InitialPoint):
            self.initial_point.ansatz = ansatz
            self.initial_point.grouped_property = driver_result
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
