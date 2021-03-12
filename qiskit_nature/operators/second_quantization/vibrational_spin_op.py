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
from typing import List, Tuple, Union, Optional

from .. import SpinOp
from ...problems.second_quantization.vibrational.vibr_to_spin_op_label_converter import \
    calc_partial_sum_modals, convert_to_spin_op_labels


class VibrationalSpinOp(SpinOp):
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
    1. Labels
    :class:`VibrationalSpinOp` accepts the notation that encodes raising (+) and lowering (-)
    operators together with indices of modes and modals that they act on, e.g. "+_{mode_index}*{
    modal_index}". Each modal can be excited at most once.
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
        r"^([\+\-]_\d+\*\d+\s)*[\+\-]_\d+\*\d+(?!\s)$|^[\+\-]+$")

    def __init__(self, data: Union[
        List[Tuple[str, complex]],
    ], num_modes: int, num_modals: Union[int, List[int]],
                 spin: Union[float, Fraction] = Fraction(1, 2)):
        r"""
        Args:
            data: list of labels and coefficients. See the label section in
                  the documentation of :class:`VibrationalSpinOp` for more details.
            num_modes : number of modes.
            num_modals: number of modals.
            spin: positive half-integer (integer or half-odd-integer) that represents spin.
        Raises:
            ValueError: invalid data is given.
            QiskitNatureError: invalid spin value.
        """

        if isinstance(data, list):
            invalid_labels = [label for label, _ in data if self._VALID_LABEL_PATTERN.match(label)]
            if not invalid_labels:
                raise ValueError(f"Invalid labels: {invalid_labels}")

        self._vibrational_data = data
        self._num_modals = num_modals
        self._num_modes = num_modes

        if not self._is_num_modals_valid():
            raise ValueError("num_modes does not agree with the size of num_modals")
        if not self._is_labels_valid():
            raise ValueError(
                "Provided labels are not valid - indexing out of range or non-matching raising "
                "and lowering operators per mode in a term")

        self._partial_sum_modals = calc_partial_sum_modals(self._num_modes, self._num_modals)

        super().__init__(convert_to_spin_op_labels(self._vibrational_data), spin)

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

    def compose(self, other):
        # TODO: implement
        raise NotImplementedError

    def _is_num_modals_valid(self):
        if type(self.num_modals) == list and len(self.num_modals) != self.num_modes:
            return False
        return True

    def _is_labels_valid(self):
        for labels, coeff in self._vibrational_data:
            coeff_labels_split = labels.split(" ")
            check_list = [0] * self.num_modes
            for label in coeff_labels_split:
                op, mode_index, modal_index = re.split('[*_]', label)
                if mode_index >= self.num_modes or modal_index >= self.num_modals:
                    return False
                increment = 1 if op == "+" else -1
                check_list[mode_index] += increment
            if not all(v == 0 for v in check_list):
                return False
        return True
