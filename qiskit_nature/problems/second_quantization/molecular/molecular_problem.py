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
from typing import cast, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.opflow import PauliSumOp

from qiskit_nature.drivers import FermionicDriver, QMolecule
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.results import EigenstateResult, ElectronicStructureResult, DipoleTuple
from qiskit_nature.transformers import BaseTransformer
from .aux_fermionic_ops_builder import _create_all_aux_operators
from .fermionic_op_builder import build_fermionic_op
from .hopping_ops_builder import build_hopping_operators
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

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations
        provided.

        Returns:
            A list of `SecondQuantizedOp` in the following order: electronic operator,
            total magnetization operator, total angular momentum operator, total particle number
            operator, and (if available) x, y, z dipole operators.
        """
        self._molecule_data = cast(QMolecule, self.driver.run())
        self._molecule_data_transformed = cast(QMolecule, self._transform(self._molecule_data))

        electronic_fermionic_op = build_fermionic_op(self._molecule_data_transformed)
        second_quantized_ops_list = [electronic_fermionic_op] + _create_all_aux_operators(
            self._molecule_data_transformed)

        return second_quantized_ops_list

    def hopping_ops(self, qubit_converter: QubitConverter,
                    excitations: Union[str, int, List[int],
                                       Callable[[int, Tuple[int, int]],
                                                List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]
                                       ] = 'sd',
                    ) -> Tuple[Dict[str, PauliSumOp], Dict[str, List[bool]],
                               Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
        """Generates the hopping operators and their commutativity information for the specified set
        of excitations.

        Args:
            qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The
                             Z2 symmetries stored in this instance are the basis for the
                             commutativity information returned by this method.
            excitations: the types of excitations to consider. The simple cases for this input are:
                - a `str` containing any of the following characters: `s`, `d`, `t` or `q`.
                - a single, positive `int` denoting the excitation type (1 == `s`, etc.).
                - a list of positive integers.
                - and finally a callable which can be used to specify a custom list of excitations.
                  For more details on how to write such a function refer to the default method,
                  :meth:`generate_fermionic_excitations`.

        Returns:
            A tuple containing the hopping operators, the types of commutativities and the
            excitation indices.
        """
        q_molecule = cast(QMolecule, self.molecule_data)
        return build_hopping_operators(q_molecule, qubit_converter, excitations)

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

        q_molecule = cast(QMolecule, self._molecule_data)
        q_molecule_transformed = cast(QMolecule, self._molecule_data_transformed)

        result = ElectronicStructureResult()
        result.combine(eigenstate_result)
        result.computed_energies = np.asarray([e.real for e in eigenstate_result.eigenenergies])
        result.hartree_fock_energy = q_molecule.hf_energy
        result.nuclear_repulsion_energy = q_molecule.nuclear_repulsion_energy
        if q_molecule.nuclear_dipole_moment is not None:
            dipole_tuple = tuple(x for x in q_molecule.nuclear_dipole_moment)
            result.nuclear_dipole_moment = cast(DipoleTuple, dipole_tuple)
        result.ph_extracted_energy = q_molecule_transformed.energy_shift.get(
            "ParticleHoleTransformer", 0)
        result.frozen_extracted_energy = q_molecule_transformed.energy_shift.get(
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
                        aux_op_eigenvalues) >= 6 and q_molecule.has_dipole_integrals:
                    # check if the names match
                    # extract dipole moment in each axis
                    dipole_moment = []
                    for moment in aux_op_eigenvalues[3:6]:
                        if moment is not None:
                            dipole_moment += [moment[0].real]  # type: ignore
                        else:
                            dipole_moment += [None]

                    result.reverse_dipole_sign = q_molecule.reverse_dipole_sign
                    result.computed_dipole_moment.append(cast(DipoleTuple,
                                                              tuple(dipole_moment)))
                    result.ph_extracted_dipole_moment.append(
                        (q_molecule_transformed.x_dip_energy_shift.get(
                            "ParticleHoleTransformer", 0),
                         q_molecule_transformed.y_dip_energy_shift.get(
                             "ParticleHoleTransformer", 0),
                         q_molecule_transformed.z_dip_energy_shift.get(
                             "ParticleHoleTransformer", 0)))

                    result.frozen_extracted_dipole_moment.append(
                        (q_molecule_transformed.x_dip_energy_shift.get(
                            "FreezeCoreTransformer", 0),
                         q_molecule_transformed.y_dip_energy_shift.get(
                             "FreezeCoreTransformer", 0),
                         q_molecule_transformed.z_dip_energy_shift.get(
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
            q_molecule_transformed = cast(QMolecule, self._molecule_data_transformed)
            return np.isclose(
                q_molecule_transformed.num_alpha + q_molecule_transformed.num_beta,
                num_particles_aux) and np.isclose(0., total_angular_momentum_aux)

        return partial(filter_criterion, self)
