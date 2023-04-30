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

"""The NumPy minimum eigensolver factory for ground state calculation algorithms."""

from __future__ import annotations

from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolver, NumPyMinimumEigensolver

from qiskit_nature.deprecation import DeprecatedType, warn_deprecated
from qiskit_nature.second_q.mappers import QubitConverter, QubitMapper
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.deprecation import deprecate_arguments

from .minimum_eigensolver_factory import MinimumEigensolverFactory


class NumPyMinimumEigensolverFactory(MinimumEigensolverFactory):
    """DEPRECATED A factory to construct a NumPyMinimumEigensolver.

    .. warning::

        This class is deprecated! Please see :ref:`this guide <how-to-numpy-min>` for how to replace
        your usage of it!
    """

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
        self._use_default_filter_criterion = use_default_filter_criterion
        self._minimum_eigensolver = NumPyMinimumEigensolver(**kwargs)

    @deprecate_arguments(
        "0.6.0",
        {"qubit_converter": "qubit_mapper"},
        additional_msg=(
            ". Additionally, the QubitConverter type in the qubit_mapper argument is deprecated "
            "and support for it will be removed together with the qubit_converter argument."
        ),
    )
    def get_solver(
        self,
        problem: BaseProblem,
        qubit_mapper: QubitConverter | QubitMapper,
        *,
        qubit_converter: QubitConverter | QubitMapper | None = None,
    ) -> MinimumEigensolver:
        # pylint: disable=unused-argument
        """Returns a NumPyMinimumEigensolver which possibly uses the default filter criterion
        provided by the ``problem``.

        Args:
            problem: A class encoding a problem to be solved.
            qubit_mapper: A class that converts second quantized operator to qubit operator.
                Providing a ``QubitConverter`` instance here is deprecated.
            qubit_converter: DEPRECATED A class that converts second quantized operator to qubit
                operator according to a mapper it is initialized with.
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
