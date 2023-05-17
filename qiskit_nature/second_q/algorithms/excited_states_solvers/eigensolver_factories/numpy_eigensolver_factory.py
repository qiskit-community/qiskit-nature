# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The numpy eigensolver factory for ground+excited states calculation algorithms."""

from typing import Optional, Union, List, Callable

import numpy as np
from qiskit.algorithms.eigensolvers import Eigensolver, NumPyEigensolver
from qiskit.utils.validation import validate_min

from qiskit_nature.deprecation import DeprecatedType, warn_deprecated
from qiskit_nature.second_q.problems.base_problem import BaseProblem
from .eigensolver_factory import EigensolverFactory


class NumPyEigensolverFactory(EigensolverFactory):
    """DEPRECATED A factory to construct a NumPyEigensolver.

    .. warning::

        This class is deprecated! Please see :ref:`this guide <how-to-numpy>` for how to replace
        your usage of it!
    """

    def __init__(
        self,
        filter_criterion: Callable[
            [Union[List, np.ndarray], float, Optional[List[float]]], bool
        ] = None,
        k: int = 100,
        use_default_filter_criterion: bool = False,
    ) -> None:
        """
        Args:
            filter_criterion: Callable that allows to filter eigenvalues/eigenstates. The minimum
                eigensolver is only searching over feasible states and returns an eigenstate that
                has the smallest eigenvalue among feasible states. The callable has the signature
                ``filter(eigenstate, eigenvalue, aux_values)`` and must return a boolean to indicate
                whether to consider this value or not. If there is no
                feasible element, the result can even be empty.
            use_default_filter_criterion: Whether to use default filter criteria or not.
            k: How many eigenvalues are to be computed, has a min. value of 1.
            use_default_filter_criterion: Whether to use the transformation's default filter
                criterion if ``filter_criterion`` is ``None``.
        """
        warn_deprecated(
            "0.6.0",
            DeprecatedType.CLASS,
            "NumPyMinimumEigensolverFactory",
            additional_msg=(
                ". This class is deprecated without replacement. Instead, refer to this how-to "
                "guide which explains the steps you need to take to replace the use of this class: "
                "https://qiskit.org/documentation/nature/howtos/numpy_minimum_eigensolver.html"
            ),
        )
        self._filter_criterion = filter_criterion
        self._k = k  # pylint:disable=invalid-name
        self._use_default_filter_criterion = use_default_filter_criterion

    @property
    def filter_criterion(
        self,
    ) -> Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]:
        """Returns filter criterion."""
        return self._filter_criterion

    @filter_criterion.setter
    def filter_criterion(
        self,
        value: Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool],
    ) -> None:
        """Sets filter criterion."""
        self._filter_criterion = value

    @property
    def k(self) -> int:
        """Returns k (number of eigenvalues requested)."""
        return self._k

    @k.setter
    def k(self, k: int) -> None:
        """Sets k (number of eigenvalues requested)."""
        validate_min("k", k, 1)
        self._k = k

    @property
    def use_default_filter_criterion(self) -> bool:
        """Returns whether to use the default filter criterion."""
        return self._use_default_filter_criterion

    @use_default_filter_criterion.setter
    def use_default_filter_criterion(self, value: bool) -> None:
        """Sets whether to use the default filter criterion."""
        self._use_default_filter_criterion = value

    def get_solver(self, problem: BaseProblem) -> Eigensolver:
        """Returns a ``NumPyEigensolver`` with the desired filter.

        Args:
            problem: A class encoding a problem to be solved.

        Returns:
            A ``NumPyEigensolver`` suitable to compute the ground state of the provided ``problem``.
        """
        filter_criterion = self._filter_criterion
        if not filter_criterion and self._use_default_filter_criterion:
            filter_criterion = problem.get_default_filter_criterion()

        npe = NumPyEigensolver(filter_criterion=filter_criterion, k=self.k)
        return npe
