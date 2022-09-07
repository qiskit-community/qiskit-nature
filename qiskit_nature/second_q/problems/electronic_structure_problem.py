# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
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

from functools import partial
from typing import cast, Callable, List, Optional, Tuple, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.opflow.primitive_ops import Z2Symmetries

from qiskit_nature.second_q.circuit.library.initial_states.hartree_fock import (
    hartree_fock_bitstring_mapped,
)
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.properties.bases import ElectronicBasisTransform

from .electronic_structure_result import ElectronicStructureResult
from .electronic_properties_container import ElectronicPropertiesContainer
from .eigenstate_result import EigenstateResult

from .base_problem import BaseProblem


class ElectronicStructureProblem(BaseProblem):
    """The Electronic Structure Problem.

    The attributes `num_particles` and `num_spin_orbitals` are only available _after_ the
    `second_q_ops()` method has been called! Note, that if you do so, the method will be executed
    again when the problem is being solved.

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
        from qiskit_nature.second_q.algorithms import NumPyEigensolverFactory

        expected_spin = 2
        expected_num_electrons = 6

        def filter_criterion_custom(eigenstate, eigenvalue, aux_values):
            num_particles_aux = aux_values["ParticleNumber"][0]
            total_angular_momentum_aux = aux_values["AngularMomentum"][0]

            return (
                np.isclose(expected_spin, total_angular_momentum_aux) and
                np.isclose(expected_num_electrons, num_particles_aux)
            )

        solver = NumPyEigensolverFactory(filter_criterion=filter_criterion_spin)

    """

    def __init__(self, hamiltonian: ElectronicEnergy) -> None:
        """
        Args:
            hamiltonian: the Hamiltonian of this problem.

        Raises:
            TypeError: if the provided ``hamiltonian`` is not of type :class:`.ElectronicEnergy`.
        """
        super().__init__(hamiltonian)
        self.properties: ElectronicPropertiesContainer = ElectronicPropertiesContainer()
        self.molecule: MoleculeInfo = None
        self.basis_transform: ElectronicBasisTransform = None
        # TODO: further refactoring:
        # - remove basis_transform
        # - store basis on Problem instead of in nested hamiltonian/properties
        # - store data on Problem instead of in nested hamiltonian/properties
        #   - orbital energies
        #   - orbital occupations
        #   - reference energy
        #   - number of particles
        #   - system size (number of orbitals)
        #   - overlap matrix (for future extension to generalized eigenvalue problem)

    @property
    def hamiltonian(self) -> ElectronicEnergy:
        return cast(ElectronicEnergy, self._hamiltonian)

    @property
    def num_particles(self) -> Tuple[int, int]:
        return self.properties.particle_number.num_particles

    @property
    def num_spin_orbitals(self) -> int:
        """Returns the number of spin orbitals."""
        return self.properties.particle_number.num_spin_orbitals

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> ElectronicStructureResult:
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An electronic structure result.
        """
        eigenstate_result = super().interpret(raw_result)
        result = ElectronicStructureResult()
        result.combine(eigenstate_result)
        self.hamiltonian.interpret(result)
        for prop in self.properties:
            if hasattr(prop, "interpret"):
                prop.interpret(result)  # type: ignore[attr-defined]
        result.computed_energies = np.asarray([e.real for e in eigenstate_result.eigenenergies])
        return result

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        qiskit.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.

        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """

        # pylint: disable=unused-argument
        def filter_criterion(self, eigenstate, eigenvalue, aux_values):
            num_particles_aux = aux_values["ParticleNumber"][0]
            total_angular_momentum_aux = aux_values["AngularMomentum"][0]
            particle_number = self.properties.particle_number
            return np.isclose(
                particle_number.num_alpha + particle_number.num_beta,
                num_particles_aux,
            ) and np.isclose(0.0, total_angular_momentum_aux)

        return partial(filter_criterion, self)

    def symmetry_sector_locator(
        self,
        z2_symmetries: Z2Symmetries,
        converter: QubitConverter,
    ) -> Optional[List[int]]:
        """Given the detected Z2Symmetries this determines the correct sector of the tapered
        operator that contains the ground state we need and returns that information.

        Args:
            z2_symmetries: the z2 symmetries object.
            converter: the qubit converter instance used for the operator conversion that
                symmetries are to be determined for.

        Returns:
            The sector of the tapered operators with the problem solution.
        """
        # We need the HF bitstring mapped to the qubit space but without any tapering done
        # by the converter (just qubit mapping and any two qubit reduction) since we are
        # going to determine the tapering sector
        hf_bitstr = hartree_fock_bitstring_mapped(
            num_spin_orbitals=self.num_spin_orbitals,
            num_particles=self.num_particles,
            qubit_converter=converter,
            match_convert=False,
        )
        sector = ElectronicStructureProblem._pick_sector(z2_symmetries, hf_bitstr)

        return sector

    @staticmethod
    def _pick_sector(z2_symmetries: Z2Symmetries, hf_str: List[bool]) -> List[int]:
        # Finding all the symmetries using the find_Z2_symmetries:
        taper_coeff: List[int] = []
        for sym in z2_symmetries.symmetries:
            coeff = -1 if np.logical_xor.reduce(np.logical_and(sym.z, hf_str)) else 1
            taper_coeff.append(coeff)

        return taper_coeff
