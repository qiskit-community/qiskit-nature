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
"""The Vibrational Problem class."""

from functools import partial
from typing import cast, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.opflow import PauliSumOp

from qiskit_nature.drivers import BosonicDriver, WatsonHamiltonian
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.problems.second_quantization.base_problem import BaseProblem
from qiskit_nature.problems.second_quantization.vibrational.hopping_ops_builder import \
    build_hopping_operators
from qiskit_nature.problems.second_quantization.vibrational.vibrational_op_builder import \
    build_vibrational_op
from qiskit_nature.results import EigenstateResult, VibronicStructureResult
from qiskit_nature.transformers import BaseTransformer
from .aux_vibrational_ops_builder import create_all_aux_operators


class VibrationalProblem(BaseProblem):
    """Vibrational Problem"""

    def __init__(self, bosonic_driver: BosonicDriver, num_modals: Union[int, List[int]],
                 truncation_order: int, transformers: Optional[List[BaseTransformer]] = None):
        """
        Args:
            bosonic_driver: A bosonic driver encoding the molecule information.
            transformers: A list of transformations to be applied to the molecule.
            num_modals: the number of modals per mode.
            truncation_order: order at which an n-body expansion is truncated
        """
        super().__init__(bosonic_driver, transformers)
        self.num_modals = num_modals
        self.truncation_order = truncation_order

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations
        provided.

        Returns:
            A list of `SecondQuantizedOp` in the following order: ... .
        """
        self._molecule_data: WatsonHamiltonian = cast(WatsonHamiltonian, self.driver.run())
        self._molecule_data_transformed: WatsonHamiltonian = \
            cast(WatsonHamiltonian, self._transform(self._molecule_data))

        vibrational_spin_op = build_vibrational_op(self._molecule_data_transformed,
                                                   self.num_modals,
                                                   self.truncation_order)

        num_modes = self._molecule_data_transformed.num_modes
        if isinstance(self.num_modals, int):
            num_modals = [self.num_modals] * num_modes
        else:
            num_modals = self.num_modals

        second_quantized_ops_list = [vibrational_spin_op] + create_all_aux_operators(num_modals)

        return second_quantized_ops_list

    def hopping_ops(self, qubit_converter: QubitConverter,
                    excitations: Union[str, int, List[int],
                                       Callable[[int, Tuple[int, int]],
                                                List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]
                                       ] = 'sd',
                    ) -> Tuple[Dict[str, PauliSumOp], Dict[str, List[bool]],
                               Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]]:

        if isinstance(self.num_modals, int):
            num_modals = [self.num_modals] * self._molecule_data_transformed.num_modes
        else:
            num_modals = self.num_modals

        return build_hopping_operators(num_modals, qubit_converter, excitations)

    def interpret(self, raw_result: Union[EigenstateResult, EigensolverResult,
                                          MinimumEigensolverResult]) -> VibronicStructureResult:
        """Interprets an EigenstateResult in the context of this transformation.
               Args:
                   raw_result: an eigenstate result object.
               Returns:
                   An vibronic structure result.
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
            eigenstate_result.aux_operator_eigenvalues = raw_result.aux_operator_eigenvalues

        result = VibronicStructureResult()
        result.combine(eigenstate_result)
        result.computed_vibronic_energies = eigenstate_result.eigenenergies
        if result.aux_operator_eigenvalues is not None:
            if not isinstance(result.aux_operator_eigenvalues, list):
                aux_operator_eigenvalues = [result.aux_operator_eigenvalues]
            else:
                aux_operator_eigenvalues = result.aux_operator_eigenvalues  # type: ignore

            result.num_occupied_modals_per_mode = []
            for aux_op_eigenvalues in aux_operator_eigenvalues:
                occ_modals = []
                for mode in range(self._molecule_data.num_modes):
                    if aux_op_eigenvalues[mode] is not None:
                        occ_modals.append(aux_op_eigenvalues[mode][0].real)  # type: ignore
                    else:
                        occ_modals.append(None)
                result.num_occupied_modals_per_mode.append(occ_modals)  # type: ignore

        return result

    def get_default_filter_criterion(self) -> Optional[Callable[[Union[List, np.ndarray], float,
                                                                 Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        aqua.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.
        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """

        # pylint: disable=unused-argument
        def filter_criterion(self, eigenstate, eigenvalue, aux_values):
            # the first num_modes aux_value is the evaluated number of particles for the given mode
            for mode in range(self.molecule_data.num_modes):
                if aux_values is None or not np.isclose(aux_values[mode][0], 1):
                    return False
            return True

        return partial(filter_criterion, self)
