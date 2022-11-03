# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The NumPy minimum eigensolver factory for ground state calculation algorithms."""

from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolver, NumPyMinimumEigensolver
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.problems import BaseProblem
from .minimum_eigensolver_factory import MinimumEigensolverFactory


class NumPyMinimumEigensolverFactory(MinimumEigensolverFactory):
    """A factory to construct a NumPyMinimumEigensolver."""

    def __init__(
        self,
        use_default_filter_criterion: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            use_default_filter_criterion: Whether to use the transformation's default filter
                criterion.
            kwargs: Keyword arguments passed to NumpyMinimumEigensolver to construct
                the internal ``MinimumEigensolver``.
        """
        self._use_default_filter_criterion = use_default_filter_criterion
        self._minimum_eigensolver = NumPyMinimumEigensolver(**kwargs)

    def get_solver(
        self, problem: BaseProblem, qubit_converter: QubitConverter
    ) -> MinimumEigensolver:
        """Returns a NumPyMinimumEigensolver which possibly uses the default filter criterion
        provided by the ``problem``.

        Args:
            problem: A class encoding a problem to be solved.
            qubit_converter: A class that converts second quantized operator to qubit operator
                             according to a mapper it is initialized with.
        Returns:
            A NumPyMinimumEigensolver suitable to compute the ground state of the molecule.
        """

        if not self.minimum_eigensolver.filter_criterion and self._use_default_filter_criterion:
            self._minimum_eigensolver.filter_criterion = problem.get_default_filter_criterion()

        return self._minimum_eigensolver

    def supports_aux_operators(self):
        return NumPyMinimumEigensolver.supports_aux_operators()

    @property
    def minimum_eigensolver(self) -> NumPyMinimumEigensolver:
        """Returns the solver instance."""
        return self._minimum_eigensolver
