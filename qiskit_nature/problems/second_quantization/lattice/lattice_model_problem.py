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
"""The Lattice Model Problem class."""

from typing import Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit_nature import ListOrDictType
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.results import EigenstateResult, LatticeModelResult
from .models.lattice_model import LatticeModel
from .. import BaseProblem


class LatticeModelProblem(BaseProblem):
    """Lattice Model Problem"""

    def __init__(
        self,
        lattice_model=LatticeModel
    ):
        super().__init__()
        self._lattice_model = lattice_model
        self._main_property_name = "LatticeEnergy"

    def second_q_ops(self) -> ListOrDictType[SecondQuantizedOp]:
        """Returns the second quantized operators created based on the lattice models.

        Returns:
            A `list` or `dict` of `SecondQuantizedOp` objects.
        """
        return [self._lattice_model.second_q_ops()]

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> LatticeModelResult:
        """Interprets an EigenstateResult in the context of this transformation.
        Args:
            raw_result: an eigenstate result object.
        Returns:
            An lattice model result.
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
        result = LatticeModelResult()
        result.combine(eigenstate_result)
        # self._grouped_property_transformed.interpret(result)
        result.computed_lattice_energies = eigenstate_result.eigenenergies
        return result


    # def get_default_filter_criterion(
    #     self,
    # ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
    #     """Returns a default filter criterion method to filter the eigenvalues computed by the
    #     eigen solver. For more information see also
    #     aqua.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.
    #     In the fermionic case the default filter ensures that the number of particles is being
    #     preserved.
    #     """

    #     # pylint: disable=unused-argument
    #     def filter_criterion(self, eigenstate, eigenvalue, aux_values):
    #         # the first num_modes aux_value is the evaluated number of particles for the given mode
    #         for mode in range(self.grouped_property_transformed.num_modes):
    #             _key = str(mode) if isinstance(aux_values, dict) else mode
    #             if aux_values is None or not np.isclose(aux_values[_key][0], 1):
    #                 return False
    #         return True

    #     return partial(filter_criterion, self)
