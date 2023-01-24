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

"""The Active-Space Reduction interface."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import cast

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy, Hamiltonian
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import BaseProblem, ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.second_q.properties import (
    AngularMomentum,
    ElectronicDensity,
    ElectronicDipoleMoment,
    Magnetization,
    ParticleNumber,
)

from .base_transformer import BaseTransformer
from .basis_transformer import BasisTransformer

LOGGER = logging.getLogger(__name__)


class ActiveSpaceTransformer(BaseTransformer):
    r"""The Active-Space reduction.

    The reduction is done by computing the inactive Fock operator which is defined as
    :math:`F^I_{pq} = h_{pq} + \sum_i 2 g_{iipq} - g_{iqpi}` and the inactive energy which is
    given by :math:`E^I = \sum_j h_{jj} + F ^I_{jj}`, where :math:`i` and :math:`j` iterate over
    the inactive orbitals.
    By using the inactive Fock operator in place of the one-electron integrals the
    description of the active space contains an effective potential generated by the inactive
    electrons. Therefore, this method permits the exclusion of non-core electrons while
    retaining a high-quality description of the system.

    For more details on the computation of the inactive Fock operator refer to
    https://arxiv.org/abs/2009.01872.

    The active space can be configured in one of the following ways through the initializer:

    - when only ``num_electrons`` and ``num_spatial_orbitals`` are specified, these integers
      indicate the number of active electrons and orbitals, respectively. The active space will
      then be chosen around the Fermi level resulting in a unique choice for any pair of
      numbers.  Nonetheless, the following criteria must be met:

        #. the remaining number of inactive electrons must be a positive, even number

        #. the number of active orbitals must not exceed the total number of orbitals minus the
           number of orbitals occupied by the inactive electrons

    - when, ``num_electrons`` is a tuple, this must indicate the number of alpha- and beta-spin
      electrons, respectively. The same requirements as listed before must be met.
    - finally, it is possible to select a custom set of active orbitals via their indices using
      ``active_orbitals``. This allows selecting an active space which is not placed around the
      Fermi level as described in the first case, above. When using this keyword argument, the
      following criteria must be met *in addition* to the ones listed above:

        #. the length of `active_orbitals` must be equal to ``num_spatial_orbitals``. Note, that
           we do **not** infer the number of active orbitals from this list of indices!

        #. the largest orbital index may **not** exceed the available ``num_spatial_orbitals``.

    References:
        - *M. Rossmannek, P. Barkoutsos, P. Ollitrault, and I. Tavernelli, arXiv:2009.01872
          (2020).*

    Attributes:
        active_basis: the :class:`.BasisTransformer` mapping from the total to the active space.
        active_density: the active electronic density.
        reference_inactive_fock: the inactive Fock operator. Setting this attribute allows you to
            enforce a custom reference operator to be used during :meth:`transform_hamiltonian`.
        reference_inactive_energy: the inactive energy. Setting this attribute allows you to enforce
            a custom reference energy to be used during :meth:`transform_hamiltonian`.
    """

    def __init__(
        self,
        num_electrons: int | tuple[int, int],
        num_spatial_orbitals: int,
        active_orbitals: list[int] | None = None,
    ):
        """
        Args:
            num_electrons: The number of active electrons. If this is a tuple, it represents the
               number of alpha- and beta-spin electrons, respectively. If this is a number, it is
               interpreted as the total number of active electrons, should be even, and implies that
               the number of alpha and beta electrons equals half of this value, respectively.
            num_spatial_orbitals: The number of active orbitals.
            active_orbitals: A list of indices specifying the spatial orbitals of the active
                space. This argument must match with the remaining arguments and should only be used
                to enforce an active space that is not chosen purely around the Fermi level.

        Raises:
            QiskitNatureError: if an invalid configuration is provided.
        """
        self._num_electrons = num_electrons
        self._num_spatial_orbitals = num_spatial_orbitals
        self._active_orbitals = active_orbitals

        try:
            self._check_configuration()
        except QiskitNatureError as exc:
            raise QiskitNatureError("Incorrect Active-Space configuration.") from exc

        self._active_orbs_indices: list[int] = None
        self.active_basis: BasisTransformer = None
        self.active_density: ElectronicIntegrals = None
        self._density_total: ElectronicIntegrals = None

        self.reference_inactive_fock: ElectronicIntegrals | None = None
        self.reference_inactive_energy: float | None = None

    def _check_configuration(self):
        if isinstance(self._num_electrons, (int, np.integer)):
            if self._num_electrons % 2 != 0:
                raise QiskitNatureError(
                    "The number of active electrons must be even! Otherwise you must specify them "
                    "as a tuple, not as:",
                    str(self._num_electrons),
                )
            if self._num_electrons < 0:
                raise QiskitNatureError(
                    "The number of active electrons cannot be negative, not:",
                    str(self._num_electrons),
                )
        elif isinstance(self._num_electrons, tuple):
            if not all(
                isinstance(n_elec, (int, np.integer)) and n_elec >= 0
                for n_elec in self._num_electrons
            ):
                raise QiskitNatureError(
                    "Neither the number of alpha, nor the number of beta electrons can be "
                    "negative, not:",
                    str(self._num_electrons),
                )
        else:
            raise QiskitNatureError(
                "The number of active electrons must be an int, or a tuple thereof, not:",
                str(self._num_electrons),
            )

        if isinstance(self._num_spatial_orbitals, (int, np.integer)):
            if self._num_spatial_orbitals < 0:
                raise QiskitNatureError(
                    "The number of active orbitals cannot be negative, not:",
                    str(self._num_spatial_orbitals),
                )
        else:
            raise QiskitNatureError(
                "The number of active orbitals must be an int, not:",
                str(self._num_spatial_orbitals),
            )

    def transform(self, problem: BaseProblem) -> BaseProblem:
        """Transforms one :class:`~qiskit_nature.second_q.problems.BaseProblem` into another.

        Args:
            problem: the problem to be transformed.

        Raises:
            NotImplementedError: when an unsupported problem type is provided.
            NotImplementedError: when the ``ElectronicStructureProblem`` is not in the
                :attr:`qiskit_nature.second_q.problems.ElectronicBasis.MO` basis.
            QiskitNatureError: If the provided ``ElectronicStructureProblem`` does not specify the
                number of particles (``num_particles``) and number of orbitals
                (``num_spatial_orbitals``), or if the amount of selected active orbital indices does
                not match the total number of active orbitals.

        Returns:
            A new `BaseProblem` instance.
        """
        if isinstance(problem, ElectronicStructureProblem):
            return self._transform_electronic_structure_problem(problem)
        else:
            raise NotImplementedError(
                f"The problem of type, {type(problem)}, is not supported by this transformer."
            )

    def _transform_electronic_structure_problem(
        self, problem: ElectronicStructureProblem
    ) -> ElectronicStructureProblem:

        if problem.basis != ElectronicBasis.MO:
            raise NotImplementedError(
                f"Transformation of an ElectronicStructureProblem in the {problem.basis} basis is "
                "not supported by this transformer. Please convert it to the ElectronicBasis.MO"
                " basis first, for example by using a BasisTransformer."
            )

        if self._active_orbs_indices is None:
            if problem.num_spatial_orbitals is None:
                raise QiskitNatureError(
                    "Using the ActiveSpaceTransformer requires the number of orbitals to be set on the "
                    "problem instance. Please set ElectronicStructureProblem.num_spatial_orbitals to "
                    "use this transformer."
                )

            if problem.num_particles is None:
                raise QiskitNatureError(
                    "Using the ActiveSpaceTransformer requires the number of particles to be set on the "
                    "problem instance. Please set ElectronicStructureProblem.num_particles to use this "
                    "transformer."
                )

            # prepare the active space
            self.prepare_active_space(
                problem.num_particles,
                problem.num_spatial_orbitals,
                occupation_alpha=problem.orbital_occupations,
                occupation_beta=problem.orbital_occupations_b,
            )

        electronic_energy = cast(ElectronicEnergy, self.transform_hamiltonian(problem.hamiltonian))

        # construct new ElectronicStructureProblem
        new_problem = ElectronicStructureProblem(electronic_energy)
        new_problem.basis = ElectronicBasis.MO
        new_problem.molecule = problem.molecule
        new_problem.reference_energy = problem.reference_energy
        new_problem.num_spatial_orbitals = self._num_spatial_orbitals

        new_problem.orbital_occupations = np.diag(self.active_density.alpha["+-"])[
            self._active_orbs_indices
        ]
        new_problem.orbital_occupations_b = np.diag(self.active_density.beta["+-"])[
            self._active_orbs_indices
        ]
        new_problem.num_particles = (
            int(sum(new_problem.orbital_occupations)),
            int(sum(new_problem.orbital_occupations_b)),
        )

        if problem.orbital_energies is not None:
            new_problem.orbital_energies = problem.orbital_energies[self._active_orbs_indices]
        if problem.orbital_energies_b is not None:
            new_problem.orbital_energies_b = problem.orbital_energies_b[self._active_orbs_indices]

        for prop in problem.properties:
            if isinstance(prop, ElectronicDipoleMoment):
                new_problem.properties.electronic_dipole_moment = (
                    self._transform_electronic_dipole_moment(prop)
                )
            elif isinstance(prop, ElectronicDensity):
                transformed = self.active_basis.transform_electronic_integrals(prop)
                new_problem.properties.electronic_density = ElectronicDensity(
                    transformed.alpha, transformed.beta, transformed.beta_alpha
                )
            elif isinstance(prop, (AngularMomentum, Magnetization, ParticleNumber)):
                new_problem.properties.add(prop.__class__(self._num_spatial_orbitals))
            else:
                LOGGER.warning("Encountered an unsupported property of type '%s'.", type(prop))

        return new_problem

    def prepare_active_space(
        self,
        total_num_electrons: int | tuple[int, int],
        total_num_spatial_orbitals: int,
        *,
        occupation_alpha: list[float] | np.ndarray | None = None,
        occupation_beta: list[float] | np.ndarray | None = None,
    ) -> None:
        """Prepares the active space.

        This method must be called manually when using this transformer on a hamiltonian outside of
        a problem instance. In all other cases, the information required here is extracted from the
        problem automatically.

        Args:
            total_num_electrons: the total number of electrons in the system represented by the
                hamiltonian which is to be transformed. If this is a tuple of integers, it encodes
                the number of alpha- and beta-spin electrons separately. Otherwise the integer value
                is assumed to indicate the sum of these two numbers.
            total_num_spatial_orbitals: the total number of spatial orbitals in the system
                represented by the hamiltonian which is to be transformed.
            occupation_alpha: the occupation of the alpha-spin orbitals. If omitted, this
                information is inferred from ``total_num_electrons`` and
                ``total_num_spatial_orbitals``.
            occupation_beta: the occupation of the beta-spin orbitals. If omitted, this
                information is inferred from ``total_num_electrons`` and
                ``total_num_spatial_orbitals``.

        Raises:
            QiskitNatureError: if any of the requirements for a valid active space configuration
                (documented in the class docstring) are not met.
        """
        if isinstance(total_num_electrons, tuple):
            num_alpha, num_beta = total_num_electrons
            sum_electrons = num_alpha + num_beta
        else:
            num_beta = total_num_electrons // 2
            num_alpha = total_num_electrons - num_beta
            sum_electrons = total_num_electrons

        if occupation_alpha is None:
            occupation_alpha = np.asarray(
                [1.0] * num_alpha + [0.0] * (total_num_spatial_orbitals - num_alpha)
            )

        if occupation_beta is None:
            occupation_beta = np.asarray(
                [1.0] * num_beta + [0.0] * (total_num_spatial_orbitals - num_beta)
            )

        self._active_orbs_indices = self._determine_active_space(
            sum_electrons, total_num_spatial_orbitals
        )

        # initialize size-reducing basis transformation
        if self.active_basis is None:
            coeff_alpha = np.zeros((total_num_spatial_orbitals, self._num_spatial_orbitals))
            coeff_alpha[self._active_orbs_indices, range(self._num_spatial_orbitals)] = 1.0
            coeff_beta = np.zeros((total_num_spatial_orbitals, self._num_spatial_orbitals))
            coeff_beta[self._active_orbs_indices, range(self._num_spatial_orbitals)] = 1.0

            self.active_basis = BasisTransformer(
                ElectronicBasis.MO,
                ElectronicBasis.MO,
                ElectronicIntegrals.from_raw_integrals(
                    coeff_alpha, h1_b=coeff_beta, validate=False
                ),
            )

        self._density_total = ElectronicIntegrals.from_raw_integrals(
            np.diag(occupation_alpha), h1_b=np.diag(occupation_beta)
        )

        if self.active_density is None:
            self.active_density = self.get_active_density_component(self._density_total)

    def get_active_density_component(
        self, total_density: ElectronicIntegrals
    ) -> ElectronicIntegrals:
        """Gets the active space density-component of the provided :class:`.ElectronicIntegrals`.

        Args:
            total_density: the density in the total orbital space.

        Returns:
            The active space component density obtained via :attr:`active_space`.
        """
        density_active = self.active_basis.transform_electronic_integrals(total_density)
        density_active.beta_alpha = None
        density_active = self.active_basis.invert().transform_electronic_integrals(density_active)
        density_active.beta_alpha = None

        return density_active

    def _determine_active_space(
        self, total_num_electrons: int, total_num_spatial_orbitals: int
    ) -> list[int]:
        """Determines the active and inactive orbital indices.

        Args:
            total_num_electrons: the total number of electrons in the system represented by the
                hamiltonian which is to be transformed. If this is a tuple of integers, it encodes
                the number of alpha- and beta-spin electrons separately. Otherwise the integer value
                is assumed to indicate the sum of these two numbers.
            total_num_spatial_orbitals: the total number of spatial orbitals in the system
                represented by the hamiltonian which is to be transformed.

        Returns:
            The list of active and inactive orbital indices.
        """
        if self._active_orbitals is not None:
            return self._active_orbitals

        if isinstance(self._num_electrons, tuple):
            num_alpha, num_beta = self._num_electrons
        elif isinstance(self._num_electrons, (int, np.integer)):
            num_alpha = num_beta = self._num_electrons // 2

        # compute number of inactive electrons
        nelec_inactive = total_num_electrons - num_alpha - num_beta

        self._validate_num_electrons(nelec_inactive)
        self._validate_num_orbitals(nelec_inactive, total_num_spatial_orbitals)

        norbs_inactive = nelec_inactive // 2
        active_orbs_idxs = list(range(norbs_inactive, norbs_inactive + self._num_spatial_orbitals))
        return active_orbs_idxs

    def _validate_num_electrons(self, nelec_inactive: int) -> None:
        """Validates the number of electrons.

        Args:
            nelec_inactive: the computed number of inactive electrons.

        Raises:
            QiskitNatureError: if the number of inactive electrons is either negative or odd.
        """
        if nelec_inactive < 0:
            raise QiskitNatureError("More electrons requested than available.")
        if nelec_inactive % 2 != 0:
            raise QiskitNatureError("The number of inactive electrons must be even.")

    def _validate_num_orbitals(self, nelec_inactive: int, num_spatial_orbitals: int) -> None:
        """Validates the number of orbitals.

        Args:
            nelec_inactive: the computed number of inactive electrons.
            num_spatial_orbitals: the total number of spatial orbitals available.

        Raises:
            QiskitNatureError: if more orbitals were requested than are available in total or if the
                               number of selected orbitals mismatches the specified number of active
                               orbitals.
        """
        if self._active_orbitals is None:
            norbs_inactive = nelec_inactive // 2
            if norbs_inactive + self._num_spatial_orbitals > num_spatial_orbitals:
                raise QiskitNatureError("More orbitals requested than available.")
        else:
            if self._num_spatial_orbitals != len(self._active_orbitals):
                raise QiskitNatureError(
                    "The number of selected active orbital indices does not "
                    "match the specified number of active orbitals."
                )
            if max(self._active_orbitals) >= num_spatial_orbitals:
                raise QiskitNatureError("More orbitals requested than available.")

    def transform_hamiltonian(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        """Transforms one :class:`~qiskit_nature.second_q.hamiltonians.Hamiltonian` into another.

        Args:
            hamiltonian: the hamiltonian to be transformed.

        Raises:
            NotImplementedError: when an unsupported hamiltonian type is provided.
            QiskitNatureError: when :meth:`prepare_active_space` was not called prior to calling
                this method.

        Returns:
            A new `Hamiltonian` instance.
        """
        if isinstance(hamiltonian, ElectronicEnergy):
            if self.active_basis is None:
                raise QiskitNatureError(
                    "In order to transform a standalone hamiltonian, you must first prepare the "
                    "active space by calling the 'prepare_active_space' method of this transformer."
                )
            return self._transform_electronic_energy(hamiltonian)
        else:
            raise NotImplementedError(
                f"The hamiltonian of type, {type(hamiltonian)}, is not supported by this "
                "transformer."
            )

    def _transform_electronic_energy(self, hamiltonian: ElectronicEnergy) -> ElectronicEnergy:
        if self.reference_inactive_fock is None:
            self.reference_inactive_fock = hamiltonian.fock(self._density_total)

        active_fock_operator = (
            hamiltonian.fock(self.active_density) - hamiltonian.electronic_integrals.one_body
        )

        inactive_fock_operator = self.reference_inactive_fock - active_fock_operator

        if self.reference_inactive_energy is None:
            reference_inactive_energy = 0.5 * ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")},
                self.reference_inactive_fock + hamiltonian.electronic_integrals.one_body,
                self._density_total,
            )
            self.reference_inactive_energy = (
                reference_inactive_energy.alpha.get("", 0.0)
                + reference_inactive_energy.beta.get("", 0.0)
                + reference_inactive_energy.beta_alpha.get("", 0.0)
            )

        e_inactive = -1.0 * ElectronicIntegrals.einsum(
            {"ij,ji": ("+-", "+-", "")}, self.reference_inactive_fock, self.active_density
        )
        e_inactive += cast(
            ElectronicIntegrals,
            0.5
            * ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")}, active_fock_operator, self.active_density
            ),
        )
        e_inactive_sum = (
            self.reference_inactive_energy
            + e_inactive.alpha.get("", 0.0)
            + e_inactive.beta.get("", 0.0)
            + e_inactive.beta_alpha.get("", 0.0)
        )

        new_hamil = ElectronicEnergy(
            self.active_basis.transform_electronic_integrals(
                inactive_fock_operator + hamiltonian.electronic_integrals.two_body
            )
        )
        new_hamil.constants = deepcopy(hamiltonian.constants)
        new_hamil.constants[self.__class__.__name__] = e_inactive_sum

        return new_hamil

    def _transform_electronic_dipole_moment(
        self, dipole_moment: ElectronicDipoleMoment
    ) -> ElectronicDipoleMoment:
        dipoles: list[ElectronicIntegrals] = []
        dip_inactive: list[float] = []
        for dipole in [dipole_moment.x_dipole, dipole_moment.y_dipole, dipole_moment.z_dipole]:
            # In the dipole case, there are no two-body terms. Thus, the inactive Fock operator
            # is unaffected by the density and equals the one-body terms.
            one_body = dipole.one_body

            e_inactive = ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")}, one_body, self._density_total
            )
            e_inactive -= ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")}, one_body, self.active_density
            )
            dipoles.append(self.active_basis.transform_electronic_integrals(one_body))
            dip_inactive.append(
                e_inactive.alpha.get("", 0.0)
                + e_inactive.beta.get("", 0.0)
                + e_inactive.beta_alpha.get("", 0.0)
            )

        new_dipole_moment = ElectronicDipoleMoment(
            x_dipole=dipoles[0],
            y_dipole=dipoles[1],
            z_dipole=dipoles[2],
        )
        new_dipole_moment.constants = deepcopy(dipole_moment.constants)
        new_dipole_moment.constants[self.__class__.__name__] = (
            dip_inactive[0],
            dip_inactive[1],
            dip_inactive[2],
        )
        new_dipole_moment.reverse_dipole_sign = dipole_moment.reverse_dipole_sign
        new_dipole_moment.nuclear_dipole_moment = dipole_moment.nuclear_dipole_moment

        return new_dipole_moment
