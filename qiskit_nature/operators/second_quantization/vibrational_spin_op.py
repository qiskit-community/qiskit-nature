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
"""A Vibrational Spin operator.
Note: this implementation differs fundamentally from the `FermionicOperator` and `BosonicOperator`
as it relies an the mathematical representation of spin matrices as (e.g.) explained in [1].
[1]: https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
"""
import re
from fractions import Fraction
from typing import Optional, List, Tuple, Union, cast

import numpy as np

from .particle_op import ParticleOp
from ... import QiskitNatureError


# TODO decide whether it inherits from SpinOp, if yes, some methods will go away from here
class VibrationalSpinOp(ParticleOp):
    """Vibrational Spin type operators.
    **Label**
    Allowed characters for primitives of labels are + and -.
    .. list-table::
        :header-rows: 1
        * - `+`
          - :math:`S_+`
          - Raising operator
        * - `-`
          - :math:`S_-`
          - Lowering operator
    1. Sparse Label (if underscore `_` exists in the label)
    # TODO explain labelling strategy when decided
    For now, accepts a friendly notation, e.g. "+_{mode_index}*{modal_index}", with a possibility
    to convert to an unfriendly index which is similar as SpinOp.
    **Initialization**
    # TODO
    **Algebra**
    :class:`VibrationalSpinOp` supports the following basic arithmetic operations: addition,
    subtraction,
    scalar multiplication, and dagger(adjoint).
    For example,
    .. jupyter-execute::
        from qiskit_nature.operators import VibrationalSpinOp
        print("Raising operator:")
        print(x + 1j * y)
        plus = SpinOp("+", spin=3/2)
        print("This is same with: ", plus)
        print("Lowering operator:")
        print(x - 1j * y)
        minus = SpinOp("-", spin=3/2)
        print("This is same with: ", minus)
        print("Dagger")
        print(~(1j * z))
    """

    _VALID_LABEL_PATTERN = re.compile(
        r"^([\+\-]_\d+\s)*[\+\-]_\d+(?!\s)$|^[\+\-]+$")

    # TODO do we want XYZ or only +-?

    def __init__(
            self,
            data: Union[
                str,
                List[Tuple[str, complex]],
                Tuple[np.ndarray, np.ndarray],
            ],
            num_modes: int, num_modals: Union[int, List[int]],
            spin: Union[float, Fraction] = Fraction(1, 2),

    ):
        r"""
        Args:
            data: label string, list of labels and coefficients. See the label section in
                  the documentation of :class:`VibrationalSpinOp` for more details.
            num_modes : number of modes
            num_modals: number of modals
            spin: positive half-integer (integer or half-odd-integer) that represents spin.
        Raises:
            ValueError: invalid data is given.
            QiskitNatureError: invalid spin value
        """
        self._coeffs: np.ndarray

        spin = Fraction(spin)
        if spin.denominator not in (1, 2):
            raise QiskitNatureError(
                f"spin must be a positive half-integer (integer or half-odd-integer), not {spin}."
            )
        self._dim = int(2 * spin + 1)

        self._num_modals = num_modals
        self._num_modes = num_modes

        if not self._is_num_modals_valid():
            raise ValueError("num_modes does not agree with the size of num_modals")

        self._partial_sum_modals = self._calc_partial_sum_modals()

        if isinstance(data, list):
            invalid_labels = [label for label, _ in data if self._VALID_LABEL_PATTERN.match(label)]
            if not invalid_labels:
                raise ValueError(f"Invalid labels: {invalid_labels}")
        self.data = data  # TODO possibly process it somehow
        # Make immutable
        self._coeffs.flags.writeable = False

    def _calc_partial_sum_modals(self):
        summed = 0
        partial_sum_modals = [0]
        if type(self.num_modals) == list:
            for mode_len in self.num_modals:
                summed += mode_len
                partial_sum_modals.append(summed)
            return partial_sum_modals
        elif type(self.num_modals) == int:
            for _ in range(self.num_modes):
                summed += self.num_modals
                partial_sum_modals.append(summed)
            return partial_sum_modals
        else:
            raise ValueError(f"num_modals of incorrect type {type(self.num_modals)}.")

    @property
    def spin(self) -> Fraction:
        """The spin number.
        Returns:
            Spin number
        """
        return Fraction(self._dim - 1, 2)

    @property
    def num_modes(self) -> int:
        """The number of modes.
        Returns:
            The number of modes
        """
        return self._num_modes

    @property
    def num_modals(self) -> Union[int, List[int]]:
        """The number of modals.
        Returns:
            The number of modals
        """
        return self._num_modals

    def _is_num_modals_valid(self):
        if type(self.num_modals) == list and len(self.num_modals) != self.num_modes:
            return False
        return True

    def _get_ind_from_mode_modal(self, mode_index, modal_index):
        return self._partial_sum_modals[mode_index] + modal_index

    def add(self, other):
        raise NotImplementedError()

    def compose(self, other):
        # TODO: implement
        raise NotImplementedError()

    def mul(self, other: complex):
        raise NotImplementedError()

    def adjoint(self):
        raise NotImplementedError()

    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None):
        raise NotImplementedError()
