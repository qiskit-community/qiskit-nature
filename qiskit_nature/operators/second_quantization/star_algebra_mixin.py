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

import copy
from abc import ABC, abstractmethod

import numpy as np


class StarAlgebraMixin(ABC):
    """The Second Quantized Operator base interface.

    This interface should be implemented by all creation- and annihilation-type particle operators
    in the second-quantized formulation.
    """

    @abstractmethod
    def __add__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __matmul__(self, other):
        raise NotImplementedError

    @abstractmethod
    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of self"""
        raise NotImplementedError

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """Overloads the division operator `/` for division by number-type objects."""
        return self * (1 / other)

    def __neg__(self):
        return -1 * self

    def __pow__(self, power):
        """Overloads the power operator `**` for applying an operator `self`, `power` number of
        times, e.g. op^{power} where `power` is a positive integer.
        """
        if isinstance(power, (int, np.integer)):
            if power < 0:
                raise UserWarning("The input `power` must be a non-negative integer")

            if power == 0:
                return self.__class__("I" * self.register_length)

            operator = copy.deepcopy(self)
            for _ in range(power - 1):
                operator @= self
            return operator

        raise TypeError(
            f"Unsupported operand type(s) for **: '{self.__class__.__name__}' and "
            "'{}'".format(type(power).__name__)
        )
