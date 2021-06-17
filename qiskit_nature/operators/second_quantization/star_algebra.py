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

"""The star algebra mixin abstract base class."""

import warnings
from abc import ABC, abstractmethod

from qiskit.quantum_info.operators.mixins import MultiplyMixin


class StarAlgebraMixin(MultiplyMixin, ABC):
    """**DEPRECATED** The star algebra mixin class.

    Star algebra is an algebra with an adjoint.

    This class overrides:

        - ``*``, ``__mul__`` -> :meth:`mul`
        - ``+``, ``__add__`` -> :meth:`add`
        - ``@``, ``__matmul__`` -> :meth:`compose`

    The following abstract methods must be implemented by subclasses:

        - :meth:``mul(self, other)``
        - :meth:``add(self, other)``
        - :meth:``compose(self, other)``
    """

    # Scalar multiplication

    @abstractmethod
    def mul(self, other: complex):
        """Return scalar multiplication of self and other, overloaded by `*`."""
        return NotImplementedError

    def __mul__(self, other: complex):
        _warn()
        return self.mul(other)

    def _multiply(self, other: complex):
        _warn()
        return self.mul(other)

    # Addition, substitution

    @abstractmethod
    def add(self, other):
        """Return Operator addition of self and other, overloaded by `+`."""
        return NotImplementedError

    def __add__(self, other):
        _warn()
        return self.add(other)

    def __radd__(self, other):
        _warn()
        if other == 0:
            return self
        return self.add(other)

    def __sub__(self, other):
        _warn()
        return self.add(-other)

    # Operator multiplication

    @abstractmethod
    def compose(self, other):
        """Overloads the matrix multiplication operator `@` for self and other.

        `Compose` computes operator composition between self and other (linear algebra-style:
        A@B(x) = A(B(x))).
        """
        raise NotImplementedError

    def __matmul__(self, other):
        _warn()
        return self.compose(other)

    def __pow__(self, power: int):
        """Overloads the power operator `**` for applying an operator `self`, `power` number of
        times, e.g. op^{power} where `power` is a positive integer.
        """
        _warn()
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

    # Adjoint

    @abstractmethod
    def adjoint(self):
        """Returns the complex conjugate transpose (dagger) of self"""
        raise NotImplementedError

    @property
    def dagger(self):
        """Alias of :meth:`adjoint()`"""
        _warn()
        return self.adjoint()

    def __invert__(self):
        """Overload unary `~` to return Operator adjoint."""
        _warn()
        return self.adjoint()


def _warn():
    warnings.warn(
        "This StarAlgebraMixin is deprecated as of 0.2.0, "
        "and will be removed no earlier than 3 months after the release. "
        "You should use the qiskit.opflow.mixins.StarAlgebraMixin "
        "as a direct replacement instead.",
        DeprecationWarning,
        stacklevel=2,
    )
