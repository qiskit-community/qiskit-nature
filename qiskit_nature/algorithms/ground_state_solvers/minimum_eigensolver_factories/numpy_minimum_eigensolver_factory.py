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

"""The numpy minimum eigensolver factory for ground state calculation algorithms."""

from typing import Optional, Union, List, Callable
import numpy as np
from qiskit.algorithms import MinimumEigensolver, NumPyMinimumEigensolver
from qiskit_nature.deprecation import (
    DeprecatedType,
    deprecate_property,
    deprecate_positional_arguments,
)
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import BaseProblem
from .minimum_eigensolver_factory import MinimumEigensolverFactory


class NumPyMinimumEigensolverFactory(MinimumEigensolverFactory):
    """A factory to construct a NumPyMinimumEigensolver."""

    @deprecate_positional_arguments(
        version="0.4",
        func_name="NumPyMinimumEigensolver Constructor",
        old_function_arguments=["self", "filter_criterion", "use_default_filter_criterion"],
        stack_level=2,
    )
    def __init__(
        self,
        use_default_filter_criterion: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            use_default_filter_criterion: whether to use the transformation's default filter
            criterion if ``filter_criterion`` is ``None``.
            kwargs: keyword arguments passed to NumpyMinimumEigensolver to construct
            the internal `MinimumEigensolver`. Note that filter_criterion is now accessed through
            NumpyMinimumEigensolver.
        """
        self._use_default_filter_criterion = use_default_filter_criterion
        self._minimum_eigensolver = NumPyMinimumEigensolver(**kwargs)

    @property  # type: ignore
    @deprecate_property(
        "0.4",
        new_type=DeprecatedType.FUNCTION,
        new_name="__init__",
    )
    def filter_criterion(
        self,
    ) -> Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]:
        """returns filter criterion"""
        return self.minimum_eigensolver.filter_criterion

    @filter_criterion.setter  # type: ignore
    @deprecate_property(
        "0.4",
        new_type=DeprecatedType.FUNCTION,
        new_name="__init__",
    )
    def filter_criterion(
        self,
        value: Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool],
    ) -> None:
        """sets filter criterion"""
        self.minimum_eigensolver.filter_criterion = value

    @property  # type: ignore
    @deprecate_property(
        "0.4",
        new_type=DeprecatedType.FUNCTION,
        new_name="__init__",
    )
    def use_default_filter_criterion(self) -> bool:
        """returns whether to use the default filter criterion"""
        return self._use_default_filter_criterion

    @use_default_filter_criterion.setter  # type: ignore
    @deprecate_property(
        "0.4",
        new_type=DeprecatedType.FUNCTION,
        new_name="__init__",
    )
    def use_default_filter_criterion(self, value: bool) -> None:
        """sets whether to use the default filter criterion"""
        self._use_default_filter_criterion = value

    def get_solver(
        self, problem: BaseProblem, qubit_converter: QubitConverter
    ) -> MinimumEigensolver:
        """Returns a NumPyMinimumEigensolver which possibly uses the default filter criterion
        provided by the ``problem``.

        Args:
            problem: a class encoding a problem to be solved.
            qubit_converter: a class that converts second quantized operator to qubit operator
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
