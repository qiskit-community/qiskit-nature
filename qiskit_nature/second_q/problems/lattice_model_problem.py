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

from typing import cast, Union

from qiskit.algorithms.eigensolvers import EigensolverResult
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolverResult
from qiskit_nature.second_q.hamiltonians import LatticeModel
from qiskit_nature.second_q.properties import Interpretable

from .base_problem import BaseProblem
from .lattice_model_result import LatticeModelResult
from .lattice_properties_container import LatticePropertiesContainer
from .eigenstate_result import EigenstateResult


class LatticeModelProblem(BaseProblem):
    """A lattice model problem.

    This class specifically deals with handling of :class:`.LatticeModel` type hamiltonians.

    The following attributes can be read and updated once the ``LatticeModelProblem`` object has
    been constructed.

    Attributes:
        properties (LatticePropertiesContainer): a container for additional observable operator
            factories.
    """

    def __init__(self, hamiltonian: LatticeModel) -> None:
        """
        Args:
            hamiltonian: A lattice model class to create second quantized operators.

        Raises:
            TypeError: if the provided ``hamiltonian`` is not of type :class:`.LatticeModel`.
        """
        super().__init__(hamiltonian)
        self.properties: LatticePropertiesContainer = LatticePropertiesContainer()

    @property
    def hamiltonian(self) -> LatticeModel:
        """Returns the hamiltonian wrapped by this problem."""
        return cast(LatticeModel, self._hamiltonian)

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
        eigenstate_result = super().interpret(raw_result)
        result = LatticeModelResult()
        result.combine(eigenstate_result)
        if isinstance(self.hamiltonian, Interpretable):
            self.hamiltonian.interpret(result)
        for prop in self.properties:
            if isinstance(prop, Interpretable):
                prop.interpret(result)
        result.computed_lattice_energies = eigenstate_result.eigenvalues
        return result
