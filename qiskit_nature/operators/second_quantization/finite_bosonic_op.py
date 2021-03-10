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

"""A finite-dimensional approximated Bosonic operator.

"""

from fractions import Fraction
from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit_nature.operators.second_quantization.particle_op import ParticleOp
from qiskit_nature.operators.second_quantization.spin_op import SpinOp


class FiniteBosonicOp(ParticleOp):
    r"""A finite-dimensional approximated Bosonic operator.
    Its internal data structure is same with :class:`SpinOp`.

    **Initialization**

    The :class:`FiniteBosonicOp` can be initialized by string or the list of tuples.
    For example,

    .. code-block:: python

        creation = FiniteBosonicOp("+", truncation_level=4)
        annihilation = FiniteBosonicOp("-", truncation_level=4)

    are creation and annihilation operator.

    :class:`FiniteBosonicOp` can be initialized with internal data structure (`numpy.ndarray`)
    directly. See the documentation of :class:`SpinOp` for more details.

    **Algebraic operations**

    :class:`FiniteBosonicOp` supports the following basic arithmetic operations: addition,
    subtraction, scalar multiplication, and dagger(adjoint).

    """

    def __init__(
        self,
        data: Union[
            str,
            List[Tuple[str, complex]],
            Tuple[np.ndarray, np.ndarray],
        ],
        truncation_level: int = 2,
    ):
        r"""
        Args:
            data: label string, list of labels and coefficients. See the label section in
                  the documentation of :class:`FiniteBosonicOp` for more details.
            truncation_level: positive integer that represents the truncation level
                              (dimension of the Hilbert space, the number of modals).

        Raises:
            ValueError: invalid data is given.
            TypeError: invalid parameter type
        """
        if not isinstance(truncation_level, int):
            raise TypeError(
                f"The parameter `truncation_level` must be int, not {type(truncation_level)}."
            )
        if truncation_level <= 0:
            raise ValueError("The parameter `truncation_level` must be positive.")

        self._data = SpinOp(data, Fraction(truncation_level - 1, 2))

    def __repr__(self) -> str:
        if len(self) == 1 and self._data._coeffs[0] == 1:
            return f"FiniteBosonicOp('{self.to_list()[0][0]}')"
        # TODO truncate
        return f"FiniteBosonicOp({self.to_list()}, truncation_level={self._data._dim})"

    def __str__(self) -> str:
        return self._data.__str__()

    def __len__(self) -> int:
        return len(self._data)

    @property
    def register_length(self) -> int:
        return self._data._register_length

    @property
    def truncation_level(self) -> int:
        """The truncation level is the dimension of the Hilbert space (the number of modal).

        Returns:
            Truncation level
        """
        return self._data._dim

    def add(self, other: "FiniteBosonicOp") -> "FiniteBosonicOp":
        if not isinstance(other, FiniteBosonicOp):
            raise TypeError(
                "Unsupported operand type(s) for +: 'FiniteBosonicOp' and "
                f"'{type(other).__name__}'"
            )

        if self.register_length != other.register_length:
            raise TypeError("Incompatible register lengths for '+'.")

        if self.truncation_level != other.truncation_level:
            raise TypeError(
                f"Addition between truncation_level {self.truncation_level} and "
                f"{other.truncation_level} is invalid."
            )

        new_data = self._data + other._data
        return FiniteBosonicOp(
            (new_data._spin_array, new_data._coeffs), truncation_level=self.truncation_level
        )

    def compose(self, other):
        # TODO: implement
        raise NotImplementedError

    def mul(self, other: complex) -> "FiniteBosonicOp":
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                "Unsupported operand type(s) for *: 'FiniteBosonicOp' and "
                f"'{type(other).__name__}'"
            )
        new_data = self._data.mul(other)

        return FiniteBosonicOp(
            (new_data._spin_array, new_data._coeffs), truncation_level=self.truncation_level
        )

    def adjoint(self) -> "FiniteBosonicOp":
        new_data = self._data.adjoint()
        return FiniteBosonicOp(
            (new_data._spin_array, new_data._coeffs), truncation_level=self.truncation_level
        )

    def reduce(
        self, atol: Optional[float] = None, rtol: Optional[float] = None
    ) -> "FiniteBosonicOp":
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        new_data = self._data.reduce(atol, rtol)
        return FiniteBosonicOp(
            (new_data._spin_array, new_data._coeffs), truncation_level=self.truncation_level
        )

    def to_list(self) -> List[Tuple[str, complex]]:
        """Getter for the list which represents `self`

        Returns:
            The list [(label, coeff)]
        """
        return self._data.to_list()

    def to_matrix(self) -> np.ndarray:
        """Convert to dense matrix

        Returns:
            The matrix (numpy.ndarray with dtype=numpy.complex128)
        """

        return self._data.to_matrix()
