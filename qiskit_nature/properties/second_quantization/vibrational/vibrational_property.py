# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Vibrational property."""

from typing import Optional

from .bases import VibrationalBasis
from ..second_quantized_property import SecondQuantizedProperty


class VibrationalProperty(SecondQuantizedProperty):
    """The Vibrational property."""

    def __init__(
        self,
        name: str,
        basis: Optional[VibrationalBasis] = None,
    ):
        """
        Args:
            basis: the ``VibrationalBasis`` through which to map the integrals into second
                quantization. This property **MUST** be set before the second-quantized operator can
                be constructed.
        """
        super().__init__(name)
        self._basis = basis

    @property
    def basis(self) -> VibrationalBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: VibrationalBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    def __repr__(self) -> str:
        string = [super().__repr__() + ":"]
        string += [f"\t{line}" for line in str(self._basis).split("\n")]
        return "\n".join(string)
