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

"""A generic Spin operator.

Note: this implementation differs fundamentally from the `FermionicOperator` and `BosonicOperator`
as it relies an the mathematical representation of spin matrices as (e.g.) explained in [1].

[1]: https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from qiskit_nature import QiskitNatureError

from .particle_op import ParticleOp


class SpinOp(ParticleOp):
    """Spin type operators. A class for products and powers of XYZ-ordered Spin operators."""

    def __init__(
            self,
            data: Union[
                List[Tuple[List[int], List[int], List[int], complex]],
                Tuple[np.ndarray, List[complex]]
                ],
            spin: float = 1/2
    ):
        """ Initialize ``SpinOp``
        Args:
            spin: positive integer or half-integer which represents spin
            data: TODO: write here

        Raises:
            QiskitNatureError: invalid data is given
        """
        # 1. Parse input
        if (round(2 * spin) != 2 * spin) or (spin <= 0):
            raise QiskitNatureError('spin must be a positive integer or half-integer')
        self._dim = int(round(2 * spin)) + 1

        # TODO: validation
        # for elem in operator_list:
        #    assert isinstance(elem, SpinOperator)
        #    assert len(elem) == self._register_length, \
        #        'Cannot sum operators acting on registers of different length'
        #    assert elem.spin == self.spin, \
        #        'Cannot sum operators with different spins.'
        if isinstance(data, tuple):
            self._spin_array = data[0]
            self._register_length = self._spin_array.shape[1]//3
            self._coeffs = data[1]

        if isinstance(data, list):
            self._register_length = len(data[0][0])
            array, self._coeffs = zip(*[(d[0] + d[1] + d[2], d[3]) for d in data])  # type: ignore
            self._spin_array = np.array(array, dtype=np.uint8)

    @property
    def register_length(self):
        return self._register_length

    @property
    def spin(self) -> float:
        """The spin number.

        Returns:
            Spin number
        """
        return (self._dim - 1) / 2

    @property
    def x(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) X operators on the spin system.
        I.e. [0, 4, 2] corresponds to X0^0 \\otimes X1^4 \\otimes X2^2, where Xi acts on the i-th
        spin system in the register.
        """
        return self._spin_array[:, 0:self.register_length]

    @property
    def y(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) Y operators on the spin system.
        I.e. [0, 4, 2] corresponds to Y0^0 \\otimes Y1^4 \\otimes Y2^2, where Yi acts on the i-th
        spin system in the register.
        """
        reg_len = self.register_length
        return self._spin_array[:, reg_len:2 * reg_len]

    @property
    def z(self) -> np.ndarray:
        """A np.ndarray storing the power i of (spin) Z operators on the spin system.
        I.e. [0, 4, 2] corresponds to Z0^0 \\otimes Z1^4 \\otimes Z2^2, where Zi acts on the i-th
        spin system in the register.
        """
        reg_len = self.register_length
        return self._spin_array[:, 2*reg_len:3*reg_len]

    def __repr__(self) -> str:
        data = [
            (x.tolist(), y.tolist(), z.tolist(), coeff)
            for x, y, z, coeff in zip(self.x, self.y, self.z, self._coeffs)
        ]
        return f"SpinOp({data})"

    def __str__(self) -> str:
        if len(self) == 1:
            label, coeff = self.to_list()[0]
            return f"{label} * {coeff}"
        return "  " + "\n+ ".join(
            [f"{label} * {coeff}" for label, coeff in self.to_list()]
        )

    def add(self, other: "SpinOp") -> "SpinOp":
        if not isinstance(other, SpinOp):
            raise TypeError("Unsupported operand type(s) for +: 'SpinOp' and "
                            f"'{type(other).__name__}'")

        if self.register_length != other.register_length:
            raise TypeError("Incompatible register lengths for '+'.")

        if self.spin != other.spin:
            raise TypeError(f"Addition between spin {self.spin} and spin {other.spin} is invalid.")

        spin_array = np.vstack([self._spin_array, other._spin_array])
        coeffs = self._coeffs + other._coeffs

        return SpinOp((spin_array, coeffs), spin=self.spin)

    def compose(self, other):
        raise NotImplementedError

    def mul(self, other: complex):
        if not isinstance(other, (int, float, complex)):
            raise TypeError("Unsupported operand type(s) for *: 'SpinOp' and "
                            f"'{type(other).__name__}'")

        coeffs = [coeff * other for coeff in self._coeffs]
        return SpinOp((self._spin_array, coeffs), spin=self.spin)

    def adjoint(self):
        # Note: X, Y, Z are hermitian, therefore the dagger operation on a SpinOperator amounts
        # to simply complex conjugating the coefficient.
        coeffs = [coeff.conjugate() for coeff in self._coeffs]
        return SpinOp((self._spin_array, coeffs), spin=self.spin)

    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None):
        raise NotImplementedError

    def _generate_label(self, i):
        """Generates the string description of `self`."""
        labels_list = []
        for pos, (n_x, n_y, n_z) in enumerate(zip(self.x[i], self.y[i], self.z[i])):
            label = []
            if n_x > 0:
                label.append(f'X^{n_x}')
            if n_y > 0:
                label.append(f'Y^{n_y}')
            if n_z > 0:
                label.append(f'Z^{n_z}')
            if not label:
                label.append(f'I')
            labels_list.append(' '.join(label) + f'[{self.register_length - pos - 1}]')
        return ' | '.join(labels_list)

    def __len__(self) -> int:
        return len(self._coeffs)

    def to_list(self) -> List[Tuple[str, complex]]:
        """Getter for the list which represents `self`

        Returns:
            The list [(label, coeff)]
        """
        return [(self._generate_label(i), self._coeffs[i]) for i in range(len(self))]
