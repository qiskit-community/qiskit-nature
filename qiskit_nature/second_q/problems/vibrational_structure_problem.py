# This code is part of a Qiskit project.
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

"""The Vibrational Structure Problem class."""

from __future__ import annotations

from functools import partial
from typing import cast, Callable, List, Optional, Union

import numpy as np

from qiskit_algorithms import EigensolverResult, MinimumEigensolverResult

from qiskit_nature.second_q.hamiltonians import VibrationalEnergy
from qiskit_nature.second_q.properties import Interpretable

from .base_problem import BaseProblem
from .vibrational_basis import VibrationalBasis
from .vibrational_structure_result import VibrationalStructureResult
from .vibrational_properties_container import VibrationalPropertiesContainer
from .eigenstate_result import EigenstateResult


class VibrationalStructureProblem(BaseProblem):
    """Vibrational Structure Problem

    The following attributes can be read and updated once the ``VibrationalStructureProblem`` object
    has been constructed.

    Attributes:
        properties (VibrationalPropertiesContainer): a container for additional observable operator
            factories.
        basis (VibrationalBasis): the second-quantization basis in which the problem's operators are
            expressed.
    """

    def __init__(self, hamiltonian: VibrationalEnergy) -> None:
        """
        Args:
            hamiltonian: the Hamiltonian of this problem.
        """
        super().__init__(hamiltonian)
        self.properties: VibrationalPropertiesContainer = VibrationalPropertiesContainer()
        self.basis: VibrationalBasis = None

    @property
    def hamiltonian(self) -> VibrationalEnergy:
        return cast(VibrationalEnergy, self._hamiltonian)

    @property
    def num_modals(self) -> list[int]:
        """The number of modals into which each mode got expanded in second-quantization."""
        return self.basis.num_modals

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> VibrationalStructureResult:
        """Interprets an EigenstateResult in the context of this problem.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            A vibrational structure result.
        """
        eigenstate_result = super().interpret(raw_result)
        result = VibrationalStructureResult()
        result.combine(eigenstate_result)
        if isinstance(self.hamiltonian, Interpretable):
            self.hamiltonian.interpret(result)
        for prop in self.properties:
            if isinstance(prop, Interpretable):
                prop.interpret(result)
        result.computed_vibrational_energies = eigenstate_result.eigenvalues
        return result

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        :meth:`~qiskit_algorithms.NumPyEigensolver.filter_criterion`.

        This particular default ensures that the occupation of every mode is (close to) 1.
        """

        # pylint: disable=unused-argument
        def filter_criterion(self, eigenstate, eigenvalue, aux_values):
            for mode, _ in enumerate(self.num_modals):
                if aux_values is None or not np.isclose(aux_values[str(mode)][0], 1):
                    return False
            return True

        return partial(filter_criterion, self)
