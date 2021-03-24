# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Molecular Problem class."""
from functools import partial
from typing import List, Tuple, Optional, cast, Union, Callable

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
import numpy as np
from qiskit_nature.drivers.qmolecule import QMolecule
from qiskit_nature.drivers import FermionicDriver
from qiskit_nature.operators import FermionicOp
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.results import EigenstateResult, ElectronicStructureResult, DipoleTuple
from qiskit_nature.transformers import BaseTransformer
from .integrals_calculators import calc_total_ang_momentum_ints
from .fermionic_op_builder import build_fermionic_op, build_ferm_op_from_ints
from .integrals_calculators import calc_total_magnetization_ints
from .integrals_calculators import calc_total_particle_num_ints
from ..base_problem import BaseProblem


class MolecularProblem(BaseProblem):
    """Molecular Problem"""

    def __init__(self, driver: FermionicDriver,
                 q_molecule_transformers: Optional[List[BaseTransformer]] = None):
        """

        Args:
            driver: A fermionic driver encoding the molecule information.
            q_molecule_transformers: A list of transformations to be applied to the molecule.
        """
        super().__init__(driver, q_molecule_transformers)
        self._q_molecule = None
        self._q_molecule_transformed = None

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations
        provided.

        Returns:
            A list of `SecondQuantizedOp` in the following order: electronic operator,
            total magnetization operator, total angular momentum operator, total particle number
            operator, and (if available) x, y, z dipole operators.
        """
        q_molecule = self.driver.run()
        self._q_molecule = q_molecule
        q_molecule_transformed = self._transform(q_molecule)
        self._q_molecule_transformed = q_molecule_transformed
        num_modes = q_molecule_transformed.one_body_integrals.shape[0]

        electronic_fermionic_op = build_fermionic_op(q_molecule_transformed)
        total_particle_number_ferm_op = self._create_total_particle_number_operator(num_modes)
        total_angular_momentum_ferm_op = self._create_total_angular_momentum_operator(num_modes)
        total_magnetization_ferm_op = self._create_total_magnetization_operator(num_modes)

        second_quantized_ops_list = [electronic_fermionic_op,
                                     total_particle_number_ferm_op,
                                     total_angular_momentum_ferm_op,
                                     total_magnetization_ferm_op,
                                     ]

        if q_molecule_transformed.has_dipole_integrals():
            x_dipole_operator, y_dipole_operator, z_dipole_operator = self._create_dipole_operators(
                q_molecule_transformed)
            second_quantized_ops_list += [x_dipole_operator,
                                          y_dipole_operator,
                                          z_dipole_operator]

        return second_quantized_ops_list

    @staticmethod
    def _create_dipole_operators(q_molecule: QMolecule) -> \
            Tuple[FermionicOp, FermionicOp, FermionicOp]:
        x_dipole_operator = build_ferm_op_from_ints(q_molecule.x_dipole_integrals)
        y_dipole_operator = build_ferm_op_from_ints(q_molecule.y_dipole_integrals)
        z_dipole_operator = build_ferm_op_from_ints(q_molecule.z_dipole_integrals)

        return x_dipole_operator, y_dipole_operator, z_dipole_operator

    @staticmethod
    def _create_total_magnetization_operator(num_modes) -> FermionicOp:
        return build_ferm_op_from_ints(*calc_total_magnetization_ints(num_modes))

    @staticmethod
    def _create_total_angular_momentum_operator(num_modes) -> FermionicOp:
        return build_ferm_op_from_ints(*calc_total_ang_momentum_ints(num_modes))

    @staticmethod
    def _create_total_particle_number_operator(num_modes) -> FermionicOp:
        return build_ferm_op_from_ints(*calc_total_particle_num_ints(num_modes))

    # TODO refactor by decomposing and eliminate ifs
    def interpret(self, raw_result: Union[EigenstateResult, EigensolverResult,
                                          MinimumEigensolverResult]) -> ElectronicStructureResult:
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An electronic structure result.
        """
        eigenstate_result = None
        if isinstance(raw_result, EigenstateResult):
            eigenstate_result = raw_result
        elif isinstance(raw_result, EigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = raw_result.eigenvalues
            eigenstate_result.eigenstates = raw_result.eigenstates
            eigenstate_result.aux_operator_eigenvalues = raw_result.aux_operator_eigenvalues
        elif isinstance(raw_result, MinimumEigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = np.asarray([raw_result.eigenvalue])
            eigenstate_result.eigenstates = [raw_result.eigenstate]
            eigenstate_result.aux_operator_eigenvalues = [raw_result.aux_operator_eigenvalues]

        result = ElectronicStructureResult()
        result.combine(eigenstate_result)
        result.computed_energies = np.asarray([e.real for e in eigenstate_result.eigenenergies])
        result.hartree_fock_energy = self._q_molecule.hf_energy
        result.nuclear_repulsion_energy = self._q_molecule.nuclear_repulsion_energy
        if self._q_molecule.nuclear_dipole_moment is not None:
            result.nuclear_dipole_moment = tuple(x for x in self._q_molecule.nuclear_dipole_moment)
        result.ph_extracted_energy = self._q_molecule_transformed.energy_shift.get(
            "ParticleHoleTransformer", 0)
        result.frozen_extracted_energy = self._q_molecule_transformed.energy_shift.get(
            "FreezeCoreTransformer", 0)
        if result.aux_operator_eigenvalues is not None:
            # the first three values are hardcoded to number of particles, angular momentum
            # and magnetization in this order
            result.num_particles = []
            result.total_angular_momentum = []
            result.magnetization = []
            result.computed_dipole_moment = []
            result.ph_extracted_dipole_moment = []
            result.frozen_extracted_dipole_moment = []
            if not isinstance(result.aux_operator_eigenvalues, list):
                aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
            else:
                aux_operator_eigenvalues = result.aux_operator_eigenvalues  # type: ignore
            for aux_op_eigenvalues in aux_operator_eigenvalues:
                if aux_op_eigenvalues is None:
                    continue
                if aux_op_eigenvalues[0] is not None:
                    result.num_particles.append(
                        aux_op_eigenvalues[0][0].real)  # type: ignore

                if aux_op_eigenvalues[1] is not None:
                    result.total_angular_momentum.append(
                        aux_op_eigenvalues[1][0].real)  # type: ignore

                if aux_op_eigenvalues[2] is not None:
                    result.magnetization.append(aux_op_eigenvalues[2][0].real)  # type: ignore

                # the next three are hardcoded to Dipole moments, if they are set
                if len(
                        aux_op_eigenvalues) >= 6 and self._q_molecule.has_dipole_integrals:
                    # check if the names match
                    # extract dipole moment in each axis
                    dipole_moment = []
                    for moment in aux_op_eigenvalues[3:6]:
                        if moment is not None:
                            dipole_moment += [moment[0].real]  # type: ignore
                        else:
                            dipole_moment += [None]

                    result.reverse_dipole_sign = self._q_molecule.reverse_dipole_sign
                    result.computed_dipole_moment.append(cast(DipoleTuple,
                                                              tuple(dipole_moment)))
                    result.ph_extracted_dipole_moment.append(
                        (self._q_molecule_transformed.x_dip_energy_shift.get(
                            "ParticleHoleTransformer", 0),
                         self._q_molecule_transformed.y_dip_energy_shift.get(
                             "ParticleHoleTransformer", 0),
                         self._q_molecule_transformed.z_dip_energy_shift.get(
                             "ParticleHoleTransformer", 0)))

                    result.frozen_extracted_dipole_moment.append(
                        (self._q_molecule_transformed.x_dip_energy_shift.get(
                            "FreezeCoreTransformer", 0),
                         self._q_molecule_transformed.y_dip_energy_shift.get(
                             "FreezeCoreTransformer", 0),
                         self._q_molecule_transformed.z_dip_energy_shift.get(
                             "FreezeCoreTransformer", 0)))

        return result

    def get_default_filter_criterion(self) -> Optional[Callable[[Union[List, np.ndarray], float,
                                                                 Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        qiskit.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.

        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """

        # pylint: disable=unused-argument
        def filter_criterion(self, eigenstate, eigenvalue, aux_values):
            # the first aux_value is the evaluated number of particles
            num_particles_aux = aux_values[0][0]
            # the second aux_value is the total angular momentum which (for singlets) should be zero
            total_angular_momentum_aux = aux_values[1][0]
            return np.isclose(
                self._q_molecule_transformed.num_alpha + self._q_molecule_transformed.num_beta,
                num_particles_aux) and \
                   np.isclose(0., total_angular_momentum_aux)

        return partial(filter_criterion, self)
