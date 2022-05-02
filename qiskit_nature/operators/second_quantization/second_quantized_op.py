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

"""The Sum Operator base interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from qiskit.opflow.mixins import StarAlgebraMixin
from qiskit.quantum_info.operators.mixins import TolerancesMixin
from qiskit.utils.deprecation import deprecate_function


class SecondQuantizedOp(StarAlgebraMixin, TolerancesMixin, ABC):
    """The Second Quantized Operator base interface.

    This interface should be implemented by all creation- and annihilation-type particle operators
    in the second-quantized formulation.
    """

    @property
    @abstractmethod
    def register_length(self) -> int:
        """Getter for the length of the particle register that the SumOp acts on."""
        raise NotImplementedError

    def __pow__(self, power):
        if power == 0:
            return self.__class__("I" * self.register_length)

        return super().__pow__(power)

    @abstractmethod
    def simplify(self, atol: Optional[float] = None):
        """Simplify the operator.

        Merges terms with same labels and eliminates terms with coefficients close to 0.
        Returns a new operator (the original operator is not modified).

        Args:
            atol: Absolute tolerance for checking if coefficients are zero (Default: 1e-8).

        Returns:
            The simplified operator.
        """
        raise NotImplementedError

    @abstractmethod
    def to_list(self) -> list[tuple[str, complex]]:
        """Returns the operators internal contents in list-format."""
        raise NotImplementedError

    def is_hermitian(self) -> bool:
        """Checks whether the operator is hermitian"""
        return frozenset(self.simplify().to_list()) == frozenset(
            self.adjoint().simplify().to_list()
        )

    @property  # type: ignore
    # pylint: disable=bad-docstring-quotes
    @deprecate_function(
        "Using the `dagger` property is deprecated as of version 0.2.0 and will be removed no "
        "earlier than 3 months after the release date. As an alternative, use the `adjoint()` "
        "method in place of `dagger` as a replacement."
    )
    def dagger(self):
        """DEPRECATED - Alias of :meth:`adjoint()`"""
        return self.adjoint()
