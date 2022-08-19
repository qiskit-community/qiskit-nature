# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Lattice Model Problem class."""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit_nature.second_q.operators import SecondQuantizedOp

from qiskit_nature.second_q.properties.lattice_model import LatticeModel

from .base_problem import BaseProblem
from .lattice_model_result import LatticeModelResult
from .eigenstate_result import EigenstateResult


class LatticeModelProblem(BaseProblem):
    """Lattice Model Problem class to create second quantized operators from a lattice model."""

    def __init__(self, lattice_model: LatticeModel) -> None:
        """
        Args:
            lattice_model: A lattice model class to create second quantized operators.
        """
        super().__init__(main_property_name="LatticeEnergy")
        self._lattice_model = lattice_model

    def second_q_ops(self) -> tuple[SecondQuantizedOp, dict[str, SecondQuantizedOp]]:
        """Returns the second quantized operators created based on the lattice models.

        Returns:
            A tuple, with the first object being the main operator and the second being a dictionary
            of auxiliary operators.
        """
        second_q_op = self._lattice_model.second_q_ops()
        return second_q_op, {}

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> LatticeModelResult:
        """Interprets a raw result in the context of this transformation.

        Args:
            raw_result: a raw result to be interpreted

        Returns:
            A lattice model result.
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
        result.computed_lattice_energies = eigenstate_result.eigenenergies
        return result

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        return None
