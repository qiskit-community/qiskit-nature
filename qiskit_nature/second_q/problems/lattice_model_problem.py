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

from typing import Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult

from qiskit_nature.second_q.hamiltonians.lattice_model import LatticeModel

from .base_problem import BaseProblem
from .lattice_model_result import LatticeModelResult
from .lattice_properties_container import LatticePropertiesContainer
from .eigenstate_result import EigenstateResult


class LatticeModelProblem(BaseProblem):
    """Lattice Model Problem class to create second quantized operators from a lattice model."""

    def __init__(self, lattice_model: LatticeModel) -> None:
        """
        Args:
            lattice_model: A lattice model class to create second quantized operators.
        """
        super().__init__(lattice_model)
        self.properties: LatticePropertiesContainer = LatticePropertiesContainer()

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
