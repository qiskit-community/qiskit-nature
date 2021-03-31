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

"""The Electronic Structure Problem class."""
from functools import partial
from typing import cast, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.opflow import PauliSumOp
from qiskit.opflow.primitive_ops import Z2Symmetries

from qiskit_nature.drivers import FermionicDriver, QMolecule
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.results import EigenstateResult, ElectronicStructureResult
from qiskit_nature.transformers import BaseTransformer
from qiskit_nature.problems.second_quantization.electronic.builders.aux_fermionic_ops_builder \
    import _create_all_aux_operators
from qiskit_nature.problems.second_quantization.electronic.builders.fermionic_op_builder import \
    _build_fermionic_op
from qiskit_nature.problems.second_quantization.electronic.builders.hopping_ops_builder import \
    _build_qeom_hopping_ops
from qiskit_nature.circuit.library.initial_states.hartree_fock import hartree_fock_bitstring
from .result_interpreter import _interpret
from ..base_problem import BaseProblem


class ElectronicStructureProblem(BaseProblem):
    """Electronic Structure Problem"""

    def __init__(self, driver: FermionicDriver,
                 q_molecule_transformers: Optional[List[BaseTransformer]] = None):
        """

        Args:
            driver: A fermionic driver encoding the molecule information.
            q_molecule_transformers: A list of transformations to be applied to the molecule.
        """
        super().__init__(driver, q_molecule_transformers)

    @property
    def num_particles(self) -> Tuple[int, int]:
        molecule_data_transformed = cast(QMolecule, self._molecule_data_transformed)
        return (molecule_data_transformed.num_alpha, molecule_data_transformed.num_beta)

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

        electronic_fermionic_op = _build_fermionic_op(self._molecule_data_transformed)
        second_quantized_ops_list = [electronic_fermionic_op] + _create_all_aux_operators(
            self._molecule_data_transformed)

        return second_quantized_ops_list

    def hopping_qeom_ops(self, qubit_converter: QubitConverter,
                         excitations: Union[str, int, List[int],
                                            Callable[[int, Tuple[int, int]],
                                                     List[Tuple[
                                                         Tuple[int, ...], Tuple[
                                                             int, ...]]]]] = 'sd',
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
        return _build_qeom_hopping_ops(q_molecule, qubit_converter, excitations)

    def interpret(self, raw_result: Union[EigenstateResult, EigensolverResult,
                                          MinimumEigensolverResult]) -> ElectronicStructureResult:
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An electronic structure result.
        """
        q_molecule = cast(QMolecule, self.molecule_data)
        q_molecule_transformed = cast(QMolecule, self.molecule_data_transformed)
        return _interpret(q_molecule, q_molecule_transformed, raw_result)

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

    def symmetry_sector_locator(self, z2_symmetries: Z2Symmetries) -> Optional[List[int]]:
        """Given the detected Z2Symmetries can determine the correct sector of the tapered
        operators so the correct one can be returned

        Args:
            z2_symmetries: the z2 symmetries object.

        Returns:
            the sector of the tapered operators with the problem solution
        """
        q_molecule = cast(QMolecule, self._molecule_data_transformed)

        hf_bitstr = hartree_fock_bitstring(
            num_spin_orbitals=2 * q_molecule.num_molecular_orbitals,
            num_particles=self.num_particles)
        sector_locator = self._pick_sector(z2_symmetries, hf_bitstr)

        return sector_locator

    def _pick_sector(self, z2_symmetries: Z2Symmetries, hf_str: List[bool]) -> Z2Symmetries:
        # Finding all the symmetries using the find_Z2_symmetries:
        taper_coef = []
        for sym in z2_symmetries.symmetries:
            # pylint: disable=no-member
            coef = -1 if np.logical_xor.reduce(np.logical_and(sym.z[::-1], hf_str)) else 1
            taper_coef.append(coef)

        return taper_coef
