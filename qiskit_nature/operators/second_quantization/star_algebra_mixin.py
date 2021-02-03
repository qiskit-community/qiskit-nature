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

from typing import Union
from abc import ABC, abstractmethod


class StarAlgebraMixin(ABC):
    """The star algebra mixin class.

    Star algebra means algebra with adjoint (dagger).
    """

    @abstractmethod
    def mul(self, other: complex):
        """Scalar multiplication."""
        raise NotImplementedError

    def __mul__(self, other: complex):
        return self.mul(other)

    def __rmul__(self, other: complex):
        return self.mul(other)

    def __truediv__(self, other: complex):
        """Overloads the division operator `/` for division by number-type objects."""
        return self.mul(1 / other)

    def __neg__(self):
        return self.mul(-1)

    @abstractmethod
    def add(self, other):
        r""" Return Operator addition of self and other, overloaded by ``+``.
        """
        raise NotImplementedError

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.add(other.mul(-1))

    @abstractmethod
    def compose(self, other):
        r""" Return Operator Composition between self and other (linear algebra-style:
        A@B(x) = A(B(x))), overloaded by ``@``.

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering
        conventions. Meaning, X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like

            -[Y]-[X]-

        Because Qiskit prints circuits with the initial state at the left side of the circuit.
        """
        raise NotImplementedError

    def __matmul__(self, other):
        return self.compose(other)

    def __pow__(self, power: int):
        """Overloads the power operator `**` for applying an operator `self`, `power` number of
        times, e.g. op^{power} where `power` is a positive integer.
        """
        if not isinstance(power, int):
            raise TypeError(
                f"Unsupported operand type(s) for **: '{type(self).__name__}' and "
                f"'{type(power).__name__}'"
            )

        if power < 1:
            raise UserWarning("The input `power` must be a positive integer")

        res = self
        for _ in range(1, power):
            res = res.compose(self)
        return res

    @abstractmethod
    def adjoint(self):
        """Returns the complex conjugate transpose (dagger) of self"""
        raise NotImplementedError

    @property
    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of self"""
        return self.adjoint()
