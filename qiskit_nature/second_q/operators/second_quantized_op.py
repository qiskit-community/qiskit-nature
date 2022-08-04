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
from typing import Any, Optional

import numpy as np
from qiskit.opflow.mixins import StarAlgebraMixin
from qiskit.quantum_info.operators.mixins import TolerancesMixin


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
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``,
                which would be 1e-8 unless changed by the user.
        Returns:
            The simplified operator.
        """
        raise NotImplementedError

    @abstractmethod
    def to_list(self) -> list[tuple[str, complex]]:
        """Returns the operators internal contents in list-format."""
        raise NotImplementedError

    def is_hermitian(self, atol: Optional[float] = None) -> bool:
        """Checks whether the operator is hermitian.

        Args:
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``,
                which would be 1e-8 unless changed by the user.

        Returns:
            True if the operator is hermitian up to numerical tolerance, False otherwise.
        """
        return self.equiv(self.adjoint(), atol=atol)

    def equiv(self, other: Any, atol: Optional[float] = None) -> bool:
        """Checks whether this operator is approximately equal to another operator.

        Args:
            other: The operator to compare to for approximate equality.
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``,
                which would be 1e-8 unless changed by the user.

        Returns:
            True if the operators are equal up to numerical tolerance, False otherwise.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        if atol is None:
            atol = self.atol
        diff = (self - other).simplify(atol=atol)
        return all(np.isclose(coeff, 0.0, atol=atol) for _, coeff in diff.to_list())
