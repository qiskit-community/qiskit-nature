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

"""The Sum Operator base interface."""

from abc import ABC, abstractmethod

from qiskit.opflow import PauliSumOp

from .star_algebra_mixin import StarAlgebraMixin


class ParticleOp(StarAlgebraMixin, ABC):
    """The Second Quantized Operator base interface.

    This interface should be implemented by all creation- and annihilation-type particle operators
    in the second-quantized formulation.
    """

    @property
    @abstractmethod
    def register_length(self) -> int:
        """Getter for the length of the particle register that the SumOp acts on."""
        raise NotImplementedError

    @abstractmethod
    def to_opflow(self, method) -> PauliSumOp:
        """TODO"""
        raise NotImplementedError

    def __pow__(self, power):
        if power == 0:
            return self.__class__("I" * self.register_length)

        return super().__pow__(power)
