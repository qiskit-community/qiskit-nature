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

"""The Vibrational Structure Problem class."""

from __future__ import annotations

from functools import partial
from typing import cast, Callable, List, Optional, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult

from qiskit_nature.second_q.drivers import VibrationalStructureDriver
from qiskit_nature.second_q.operators import SecondQuantizedOp
from qiskit_nature.second_q.properties import (
    VibrationalStructureDriverResult,
)
from qiskit_nature.second_q.properties.bases import HarmonicBasis
from qiskit_nature.second_q.transformers.base_transformer import BaseTransformer

from .base_problem import BaseProblem

from .vibrational_structure_result import VibrationalStructureResult
from .eigenstate_result import EigenstateResult


class VibrationalStructureProblem(BaseProblem):
    """Vibrational Structure Problem"""

    def __init__(
        self,
        bosonic_driver: VibrationalStructureDriver,
        num_modals: Union[int, List[int]],
        truncation_order: int,
        transformers: Optional[List[BaseTransformer]] = None,
    ):
        """
        Args:
            bosonic_driver: a bosonic driver encoding the molecule information.
            num_modals: the number of modals per mode.
            truncation_order: order at which an n-body expansion is truncated
            transformers: a list of transformations to be applied to the driver result.
        """
        super().__init__(bosonic_driver, transformers, "VibrationalEnergy")
        self._num_modals = num_modals
        self.truncation_order = truncation_order

    @property
    def num_modals(self) -> List[int]:
        """Returns the number of modals, always expanded as a list."""
        num_modes = cast(
            VibrationalStructureDriverResult, self._grouped_property_transformed
        ).num_modes
        if isinstance(self._num_modals, int):
            num_modals = [self._num_modals] * num_modes
        else:
            num_modals = self._num_modals
        return num_modals

    def second_q_ops(self) -> tuple[SecondQuantizedOp, dict[str, SecondQuantizedOp]]:
        """Returns the second quantized operators associated with this problem.

        Returns:
            A tuple, with the first object being the main operator and the second being a dictionary
            of auxiliary operators.
        """
        driver_result = self.driver.run()

        self._grouped_property = driver_result
        self._grouped_property_transformed = self._transform(self._grouped_property)

        self._grouped_property_transformed = cast(
            VibrationalStructureDriverResult, self._grouped_property_transformed
        )

        for prop in self._grouped_property_transformed:
            if hasattr(prop, "truncation_order"):
                prop.truncation_order = self.truncation_order

        # TODO: expose this as an argument in __init__
        basis = HarmonicBasis(self.num_modals)
        self._grouped_property_transformed.basis = basis

        second_quantized_ops = self._grouped_property_transformed.second_q_ops()
        main_op = second_quantized_ops.pop(self._main_property_name)

        return main_op, second_quantized_ops

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> VibrationalStructureResult:
        """Interprets an EigenstateResult in the context of this transformation.
        Args:
            raw_result: an eigenstate result object.
        Returns:
            An vibrational structure result.
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
        result = VibrationalStructureResult()
        result.combine(eigenstate_result)
        self._grouped_property_transformed.interpret(result)
        result.computed_vibrational_energies = eigenstate_result.eigenenergies
        return result

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        aqua.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.
        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """

        # pylint: disable=unused-argument
        def filter_criterion(self, eigenstate, eigenvalue, aux_values):
            # the first num_modes aux_value is the evaluated number of particles for the given mode
            for mode in range(self.grouped_property_transformed.num_modes):
                if aux_values is None or not np.isclose(aux_values[str(mode)][0], 1):
                    return False
            return True

        return partial(filter_criterion, self)
