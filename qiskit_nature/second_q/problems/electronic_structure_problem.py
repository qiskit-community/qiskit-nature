# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Electronic Structure Problem class."""

from __future__ import annotations

import warnings
from functools import partial
from typing import cast, Callable, List, Optional, Union, TYPE_CHECKING

import numpy as np

from qiskit.algorithms.eigensolvers import EigensolverResult
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolverResult
from qiskit.opflow.primitive_ops import Z2Symmetries as OpflowZ2Symmetries
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries

from qiskit_nature.deprecation import deprecate_function, warn_deprecated_type
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.second_q.circuit.library.initial_states.hartree_fock import (
    hartree_fock_bitstring_mapped,
)
from qiskit_nature.second_q.mappers import QubitConverter, QubitMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.properties import Interpretable

from .electronic_structure_result import ElectronicStructureResult
from .electronic_properties_container import ElectronicPropertiesContainer
from .eigenstate_result import EigenstateResult

from .base_problem import BaseProblem
from .electronic_basis import ElectronicBasis

if TYPE_CHECKING:
    from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo


class ElectronicStructureProblem(BaseProblem):
    r"""The Electronic Structure Problem.

    This class represents the problem of the electronic Schrödinger equation:

    .. math::
        \hat{H_{el}}|\Psi\rangle = E_{el}|\Psi\rangle,

    where :math:`\hat{H_{el}}` is the :class:`qiskit_nature.second_q.hamiltonians.ElectronicEnergy`
    hamiltonian, :math:`\Psi` is the wave function of the system and :math:`E_{el}` is the
    eigenvalue. When passed to a :class:`qiskit_nature.second_q.algorithms.GroundStateSolver`, you
    will be solving for the ground-state energy, :math:`E_0`.

    This class has various attributes (see below) which allow you to add additional information
    about the problem which you are trying to solve, which can be used by various modules in the
    stack. For example, specifying the number of particles in the system :attr:`num_particles` is
    useful (and even required) for many components that interact with this problem instance to make
    your life easier (for example the
    :class:`qiskit_nature.second_q.transformers.ActiveSpaceTransformer`).

    In the fermionic case the default filter ensures that the number of particles is being
    preserved.

    .. note::

        The default filter_criterion assumes a singlet spin configuration. This means, that the
        number of alpha-spin electrons is equal to the number of beta-spin electrons.
        If the :class:`~qiskit_nature.second_q.properties.AngularMomentum`
        property is available, one can correctly filter a non-singlet spin configuration with a
        custom `filter_criterion` similar to the following:

    .. code-block:: python

        import numpy as np
        from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

        expected_spin = 2
        expected_num_electrons = 6

        def filter_criterion_custom(eigenstate, eigenvalue, aux_values):
            num_particles_aux = aux_values["ParticleNumber"][0]
            total_angular_momentum_aux = aux_values["AngularMomentum"][0]

            return (
                np.isclose(total_angular_momentum_aux, expected_spin) and
                np.isclose(num_particles_aux, expected_num_electrons)
            )

        solver = NumPyEigensolver()
        solver.filter_criterion = filter_criterion_custom

    The following attributes can be read and updated once the ``ElectronicStructureProblem`` object
    has been constructed.

    Attributes:
        properties (ElectronicStructureProblem): a container for additional observable operator
            factories.
        molecule (MoleculeInfo | None): a container for molecular system data.
        basis (ElectronicBasis | None): the electronic basis of all contained orbital coefficients.
        num_spatial_orbitals (int | tuple[int, int] | None): the number of spatial orbitals in the
            system.
        reference_energy (float | None): a reference energy for the ground state of the problem.
        orbital_energies (np.ndarray | None): the energy values of the alpha-spin orbitals.
        orbital_energies_b (np.ndarray | None): the energy values of the beta-spin orbitals.
    """

    def __init__(self, hamiltonian: ElectronicEnergy) -> None:
        """
        Args:
            hamiltonian: the Hamiltonian of this problem.
        """
        super().__init__(hamiltonian)
        self.properties: ElectronicPropertiesContainer = ElectronicPropertiesContainer()
        self.molecule: "MoleculeInfo" | None = None
        self.basis: ElectronicBasis | None = None
        self._num_particles: int | tuple[int, int] | None = None
        self.num_spatial_orbitals: int | None = hamiltonian.register_length
        self._orbital_occupations: np.ndarray | None = None
        self._orbital_occupations_b: np.ndarray | None = None
        self.reference_energy: float | None = None
        self.orbital_energies: np.ndarray | None = None
        self.orbital_energies_b: np.ndarray | None = None

    @property
    def hamiltonian(self) -> ElectronicEnergy:
        return cast(ElectronicEnergy, self._hamiltonian)

    @property
    def nuclear_repulsion_energy(self) -> float | None:
        """The nuclear repulsion energy.

        See :attr:`qiskit_nature.second_q.hamiltonians.ElectronicEnergy.nuclear_repulsion_energy`
        for more details.
        """
        return self.hamiltonian.nuclear_repulsion_energy

    @property
    def num_particles(self) -> tuple[int, int] | None:
        """The number of particles in alpha- and beta-spin."""
        if self._num_particles is None:
            return None
        if isinstance(self._num_particles, tuple):
            return self._num_particles
        num_beta = self._num_particles // 2
        num_alpha = self._num_particles - num_beta
        return (num_alpha, num_beta)

    @num_particles.setter
    def num_particles(self, num_particles: int | tuple[int, int] | None) -> None:
        self._num_particles = num_particles

    @property
    def num_alpha(self) -> int | None:
        """The number of alpha-spin particles."""
        if self.num_particles is None:
            return None
        return self.num_particles[0]

    @property
    def num_beta(self) -> int | None:
        """The number of beta-spin particles."""
        if self.num_particles is None:
            return None
        return self.num_particles[1]

    @property
    def num_spin_orbitals(self) -> int | None:
        """The total number of spin orbitals."""
        if self.num_spatial_orbitals is None:
            return None
        return 2 * self.num_spatial_orbitals

    @property
    def orbital_occupations(self) -> np.ndarray | None:
        """The occupations of the alpha-spin orbitals."""
        if self._orbital_occupations is not None:
            return self._orbital_occupations

        if self.basis != ElectronicBasis.MO:
            return None

        num_orbs = self.num_spatial_orbitals
        if num_orbs is None:
            return None

        num_alpha = self.num_alpha
        if num_alpha is None:
            return None

        return np.asarray([1.0] * num_alpha + [0.0] * (num_orbs - num_alpha))

    @orbital_occupations.setter
    def orbital_occupations(self, occ: np.ndarray | None) -> None:
        self._orbital_occupations = occ

    @property
    def orbital_occupations_b(self) -> np.ndarray | None:
        """The occupations of the beta-spin orbitals."""
        if self._orbital_occupations_b is not None:
            return self._orbital_occupations_b

        if self.basis != ElectronicBasis.MO:
            return None

        num_orbs = self.num_spatial_orbitals
        if num_orbs is None:
            return None

        num_beta = self.num_beta
        if num_beta is None:
            return None

        return np.asarray([1.0] * num_beta + [0.0] * (num_orbs - num_beta))

    @orbital_occupations_b.setter
    def orbital_occupations_b(self, occ: np.ndarray | None) -> None:
        self._orbital_occupations_b = occ

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> ElectronicStructureResult:
        """Interprets an EigenstateResult in the context of this problem.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An electronic structure result.
        """
        eigenstate_result = super().interpret(raw_result)
        result = ElectronicStructureResult()
        result.combine(eigenstate_result)
        if isinstance(self.hamiltonian, Interpretable):
            self.hamiltonian.interpret(result)
        for prop in self.properties:
            if isinstance(prop, Interpretable):
                prop.interpret(result)
        result.computed_energies = np.asarray([e.real for e in eigenstate_result.eigenvalues])
        if self.reference_energy is not None:
            result.hartree_fock_energy = self.reference_energy
        return result

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigensolver. For more information see also
        :class:`qiskit.algorithms.eigensolvers.NumPyEigensolver.filter_criterion`.

        This particular default ensures that the total number of particles is conserved and that the
        angular momentum (if computed) evaluates to 0.
        """

        # pylint: disable=unused-argument
        def filter_criterion(self, eigenstate, eigenvalue, aux_values):
            eval_num_particles = aux_values.get("ParticleNumber", None)
            if eval_num_particles is None:
                return True
            num_particles_close = np.isclose(eval_num_particles[0], self.num_alpha + self.num_beta)

            eval_angular_momentum = aux_values.get("AngularMomentum", None)
            if eval_angular_momentum is None:
                return num_particles_close
            angular_momentum_close = np.isclose(eval_angular_momentum[0], 0.0)

            return num_particles_close and angular_momentum_close

        return partial(filter_criterion, self)

    # pylint: disable=bad-docstring-quotes
    @deprecate_function(
        "0.6.0",
        additional_msg=(
            ". This function is deprecated because it has been removed from the public API. It is "
            "no longer necessary to be used when working directly with QubitMapper objects outside "
            "a QubitConverter because a TaperedQubitMapper can now be obtained using the new "
            "get_tapered_mapper function provided by the problem classes."
        ),
    )
    def symmetry_sector_locator(
        self,
        z2_symmetries: OpflowZ2Symmetries | Z2Symmetries,
        converter: QubitConverter | QubitMapper,
    ) -> Optional[List[int]]:
        """Given the detected Z2Symmetries this determines the correct sector of the tapered
        operator that contains the ground state we need and returns that information.

        Args:
            z2_symmetries: the z2 symmetries object.
            converter: the ``QubitConverter`` or ``QubitMapper`` instance used for the operator
                conversion that symmetries are to be determined for.

        Raises:
            QiskitNatureError: if the :attr:`num_particles` attribute is ``None``.

        Returns:
            The sector of the tapered operators with the problem solution.
        """
        if self.num_particles is None:
            raise QiskitNatureError(
                "Determining the correct symmetry sector for Z2 symmetry reduction requires the "
                "number of particles to be set on the problem instance. Please set "
                "ElectronicStructureProblem.num_particles or disable the use of Z2Symmetries to "
                "fix this."
            )

        num_particles = self.num_particles
        if not isinstance(num_particles, tuple):
            num_particles = (self.num_alpha, self.num_beta)

        # We need the HF bitstring mapped to the qubit space but without any tapering done
        # by the converter (just qubit mapping and any two qubit reduction) since we are
        # going to determine the tapering sector
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            hf_bitstr = hartree_fock_bitstring_mapped(
                num_spatial_orbitals=self.num_spatial_orbitals,
                num_particles=num_particles,
                qubit_mapper=converter,
                match_convert=False,
            )
        sector = ElectronicStructureProblem._pick_sector(z2_symmetries, hf_bitstr)

        return sector

    def _symmetry_sector_locator(
        self,
        z2_symmetries: OpflowZ2Symmetries | Z2Symmetries,
        mapper: QubitConverter | QubitMapper,
    ) -> Optional[List[int]]:
        """Given the detected Z2Symmetries this determines the correct sector of the tapered
        operator that contains the ground state we need and returns that information.

        Args:
            z2_symmetries: the z2 symmetries object.
            mapper: the ``QubitMapper`` or ``QubitConverter`` instance (use of the latter is
                deprecated) used for the operator conversion that symmetries are to be determined
                for.

        Raises:
            QiskitNatureError: if the :attr:`num_particles` attribute is ``None``.

        Returns:
            The sector of the tapered operators with the problem solution.
        """
        if self.num_particles is None:
            raise QiskitNatureError(
                "Determining the correct symmetry sector for Z2 symmetry reduction requires the "
                "number of particles to be set on the problem instance. Please set "
                "ElectronicStructureProblem.num_particles or disable the use of Z2Symmetries to "
                "fix this."
            )
        if isinstance(mapper, QubitConverter):
            warn_deprecated_type(
                "0.6.0",
                argument_name="mapper",
                old_type="QubitConverter",
                new_type="QubitMapper",
            )

        num_particles = self.num_particles
        if not isinstance(num_particles, tuple):
            num_particles = (self.num_alpha, self.num_beta)

        hf_bitstr = hartree_fock_bitstring_mapped(
            num_spatial_orbitals=self.num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
        )
        sector = ElectronicStructureProblem._pick_sector(z2_symmetries, hf_bitstr)

        return sector

    @staticmethod
    def _pick_sector(
        z2_symmetries: OpflowZ2Symmetries | Z2Symmetries, hf_str: List[bool]
    ) -> List[int]:
        # Finding all the symmetries using the find_Z2_symmetries:
        taper_coeff: List[int] = []
        for sym in z2_symmetries.symmetries:
            coeff = -1 if np.logical_xor.reduce(np.logical_and(sym.z, hf_str)) else 1
            taper_coeff.append(coeff)

        return taper_coeff
