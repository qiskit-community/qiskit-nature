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

"""The Freeze-Core Reduction interface."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.constants import PERIODIC_TABLE
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy, Hamiltonian
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import BaseProblem, ElectronicBasis, ElectronicStructureProblem

from .active_space_transformer import ActiveSpaceTransformer, _transform_electronic_energy
from .base_transformer import BaseTransformer
from .basis_transformer import BasisTransformer


LOGGER = logging.getLogger(__name__)


class FreezeCoreTransformer(BaseTransformer):
    """The Freeze-Core reduction.

    This transformation is mathematically identical to the
    :class:`~qiskit_nature.second_q.transformers.ActiveSpaceTransformer`. The difference arises in
    the user interface: while you configure the _active_ components in the other transformer, here
    you configure the _inactive_ components. For more information on the configuration options
    please refer to the input argument description further below and for more information on the
    mathematical transformation refer to the documentation of the
    :class:`~qiskit_nature.second_q.transformers.ActiveSpaceTransformer`.

    If you want to apply this transformer to a Hamiltonian outside of a Problem instance, you need
    to prepare the active space by providing the molecular system information which your Hamiltonian
    corresponds to which would normally be extracted from the Problem object. You can do this like
    so:

    .. code-block:: python

      # assuming you have the total Hamiltonian of your system available:
      total_hamiltonian = ElectronicEnergy(...)

      # now you want to apply the freeze-core reduction
      transformer = FreezeCoreTransformer()

      # since the FreezeCoreTransformer requires molecular system information,
      # you need to create that data structure like so:
      molecule = MoleculeInfo(
          symbols=["Li", "H"],
          coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.6)],
      )
      # and since the system size depends on the basis set, you need to provide
      # the total number of spatial orbitals separately:
      total_num_spatial_orbitals = 11  # e.g. the 6-31g basis

      # this allows you to prepare the active space correctly like so:
      transformer.prepare_active_space(molecule, total_num_spatial_orbitals)

      # after preparation, you can now transform only your Hamiltonian like so
      reduced_hamiltonian = transformer.transform_hamiltonian(total_hamiltonian)
    """

    def __init__(
        self,
        freeze_core: bool = True,
        remove_orbitals: list[int] | None = None,
    ):
        """Initializes a transformer which can reduce an `ElectronicStructureProblem` to a
        configured active space.

        The orbitals to be removed are specified in two ways:

            #. When ``freeze_core`` is enabled (the default), the "core" orbitals will be determined
               automatically according to ``count_core_orbitals``. These will then be made inactive
               and removed in the same fashion as in the :class:`ActiveSpaceTransformer`.
            #. Additionally, unoccupied spatial orbitals can be removed via a list of indices
               passed to ``remove_orbitals``. It is the user's responsibility to ensure that these are
               indeed unoccupied orbitals, as no checks are performed.

        If you want to remove additional occupied orbitals, please use the
        :class:`ActiveSpaceTransformer` instead.

        Args:
            freeze_core: A boolean indicating whether to remove the "core" orbitals.
            remove_orbitals: A list of indices specifying spatial orbitals which are removed.
                             No checks are performed on the nature of these orbitals, so the user
                             must make sure that these are _unoccupied_ orbitals, which can be
                             removed without taking any energy shifts into account.
        """
        self._freeze_core = freeze_core
        self._remove_orbitals = remove_orbitals

        self._active_orbs_indices: list[int] = None
        self._active_basis: BasisTransformer = None
        self._active_density: ElectronicIntegrals = None
        self._density_total: ElectronicIntegrals = None

    def transform_hamiltonian(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        """Transforms one :class:`~qiskit_nature.second_q.hamiltonians.Hamiltonian` into another.

        Args:
            hamiltonian: the hamiltonian to be transformed.

        Raises:
            NotImplementedError: when an unsupported hamiltonian type is provided.
            QiskitNatureError: when :meth:`prepare_active_space` was not called prior to calling
                this method.

        Returns:
            A new ``Hamiltonian`` instance.
        """
        if isinstance(hamiltonian, ElectronicEnergy):
            if self._active_basis is None:
                raise QiskitNatureError(
                    "In order to transform a standalone hamiltonian, you must first prepare the "
                    "active space by calling the 'prepare_active_space' method of this transformer."
                )
            return _transform_electronic_energy(
                hamiltonian,
                self._density_total,
                self._active_density,
                self._active_basis,
                self.__class__.__name__,
            )
        else:
            raise NotImplementedError(
                f"The hamiltonian of type, {type(hamiltonian)}, is not supported by this "
                "transformer."
            )

    def transform(self, problem: BaseProblem) -> BaseProblem:
        """Transforms one :class:`~qiskit_nature.second_q.problems.BaseProblem` into another.

        Args:
            problem: the problem to be transformed.

        Raises:
            NotImplementedError: when an unsupported problem type is provided.
            NotImplementedError: when the ``ElectronicStructureProblem`` is not in the
                :attr:`qiskit_nature.second_q.problems.ElectronicBasis.MO` basis.
            QiskitNatureError: If the provided ``ElectronicStructureProblem`` does not contain a
                ``molecule``, ``num_particles`` or ``num_spatial_orbitals`` attribute.

        Returns:
            A new ``BaseProblem`` instance.
        """
        if isinstance(problem, ElectronicStructureProblem):
            return self._transform_electronic_structure_problem(problem)  # type: ignore[misc]
        else:
            raise NotImplementedError(
                f"The problem of type, {type(problem)}, is not supported by this transformer."
            )

    def prepare_active_space(
        self,
        molecule: MoleculeInfo,
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
            molecule: the molecular system information to which the hamiltonian belongs. From this,
                the involved atomic species and number of electrons will be extracted.
            total_num_spatial_orbitals: the total number of spatial orbitals in the system
                represented by the hamiltonian which is to be transformed.
            occupation_alpha: the occupation of the alpha-spin orbitals. If omitted, this
                information is inferred from the required arguments.
            occupation_beta: the occupation of the beta-spin orbitals. If omitted, this
                information is inferred from the required arguments.
        """
        sum_electrons = sum(self.Z(atom) for atom in molecule.symbols)
        # NOTE: electrons are negatively charged! Thus, when we have for example a charge of +1,
        # this indicates 1 fewer electron. In contrast, a charge of -1 is 1 more electron.
        sum_electrons -= molecule.charge
        num_alpha = sum_electrons // 2 + molecule.multiplicity - 1
        num_beta = sum_electrons - num_alpha

        if occupation_alpha is None:
            occupation_alpha = np.asarray(
                [1.0] * num_alpha + [0.0] * (total_num_spatial_orbitals - num_alpha)
            )

        if occupation_beta is None:
            occupation_beta = np.asarray(
                [1.0] * num_beta + [0.0] * (total_num_spatial_orbitals - num_beta)
            )

        self._active_orbs_indices = self._determine_active_space(
            molecule, total_num_spatial_orbitals
        )
        active_num_spatial_orbitals = len(self._active_orbs_indices)

        coeff_alpha = np.zeros((total_num_spatial_orbitals, active_num_spatial_orbitals))
        coeff_alpha[self._active_orbs_indices, range(active_num_spatial_orbitals)] = 1.0
        coeff_beta = np.zeros((total_num_spatial_orbitals, active_num_spatial_orbitals))
        coeff_beta[self._active_orbs_indices, range(active_num_spatial_orbitals)] = 1.0

        self._active_basis = BasisTransformer(
            ElectronicBasis.MO,
            ElectronicBasis.MO,
            ElectronicIntegrals.from_raw_integrals(coeff_alpha, h1_b=coeff_beta, validate=False),
        )

        self._density_total = ElectronicIntegrals.from_raw_integrals(
            np.diag(occupation_alpha), h1_b=np.diag(occupation_beta)
        )

        self._active_density = self._get_active_density_component(self._density_total)

    def _prepare_problem(self, problem: ElectronicStructureProblem) -> None:
        if problem.basis != ElectronicBasis.MO:
            raise NotImplementedError(
                f"Transformation of an ElectronicStructureProblem in the {problem.basis} basis is "
                "not supported by this transformer. Please convert it to the ElectronicBasis.MO"
                " basis first, for example by using a BasisTransformer."
            )

        if problem.num_spatial_orbitals is None:
            raise QiskitNatureError(
                "Using the FreezeCoreTransformer requires the number of orbitals to be set on the "
                "problem instance. Please set ElectronicStructureProblem.num_spatial_orbitals to "
                "use this transformer."
            )

        if problem.molecule is None:
            raise QiskitNatureError(
                "Using the FreezeCoreTransformer requires the molecule to be specified by the "
                "problem instance. Please set ElectronicStructureProblem.molecule to use this "
                "transformer."
            )

        self.prepare_active_space(problem.molecule, problem.num_spatial_orbitals)

    _transform_electronic_structure_problem = (  # pylint: disable=invalid-name
        ActiveSpaceTransformer._transform_electronic_structure_problem
    )

    def _determine_active_space(
        self, molecule: MoleculeInfo, total_num_spatial_orbitals: int
    ) -> list[int]:
        """Determines the active and inactive orbital indices.

        Args:
            molecule: the molecular system information.
            total_num_spatial_orbitals: the total number of spatial orbitals in the system.

        Returns:
            The list of active orbital indices.
        """
        inactive_orbs_idxs: list[int] = []
        if self._freeze_core:
            inactive_orbs_idxs.extend(range(self.count_core_orbitals(molecule.symbols)))
        if self._remove_orbitals is not None:
            inactive_orbs_idxs.extend(self._remove_orbitals)
        active_orbs_idxs = [
            o for o in range(total_num_spatial_orbitals) if o not in inactive_orbs_idxs
        ]
        return active_orbs_idxs

    def _get_active_density_component(
        self, total_density: ElectronicIntegrals
    ) -> ElectronicIntegrals:
        """Gets the active space density-component of the provided :class:`.ElectronicIntegrals`.

        Args:
            total_density: the density in the total orbital space.

        Returns:
            The active space component density obtained via :attr:`active_space`.
        """
        density_active = self._active_basis.transform_electronic_integrals(total_density)
        density_active.beta_alpha = None
        density_active = self._active_basis.invert().transform_electronic_integrals(density_active)
        density_active.beta_alpha = None

        return density_active

    def count_core_orbitals(self, atoms: Sequence[str]) -> int:
        """Counts the number of core orbitals in a list of atoms.

        Args:
            atoms: the list of atoms.

        Returns:
            The number of core orbitals.
        """
        count = 0
        for atom in atoms:
            z = self.Z(atom)
            if z > 2:
                count += 1
            if z > 10:
                count += 4
            if z > 18:
                count += 4
            if z > 36:
                count += 9
            if z > 54:
                count += 9
            if z > 86:
                count += 16
        return count

    def Z(self, atom: str) -> int:  # pylint: disable=invalid-name
        """Atomic Number (Z) of an atom.

        Args:
            atom: the atom kind (symbol) whose atomic number to return.

        Returns:
            The atomic number of the queried atom kind.
        """
        return PERIODIC_TABLE.index(atom.lower().capitalize())
