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
from fractions import Fraction
from typing import List, Tuple, Union

from .. import SpinOp
from ...problems.second_quantization.vibrational.vibr_to_spin_op_label_converter import \
    convert_to_spin_op_labels
from ...problems.second_quantization.vibrational.vibrational_labels_validator import \
    validate_vibrational_labels


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
    """

    def __init__(self, data: Union[List[Tuple[str, float]]], num_modes: int,
                 num_modals: Union[int, List[int]],
                 spin: Union[float, Fraction] = Fraction(1, 2)):
        r"""
        Args:
            data: list of labels and coefficients. See the label section in
                  the documentation of :class:`VibrationalSpinOp` for more details.
            num_modes : number of modes.
            num_modals: number of modals - described by a list of integers where each integer
                        describes the number of modals in a corresponding mode; in case of the
                        same number of modals in each mode it is enough to provide an integer
                        that describes the number of them.
            spin: positive half-integer (integer or half-odd-integer) that represents spin.
        Raises:
            ValueError: invalid data is given.
            QiskitNatureError: invalid spin value.
        """
        validate_vibrational_labels(data, num_modes, num_modals)

        self._vibrational_data = data
        self._num_modals = num_modals
        self._num_modes = num_modes

        self._spin_op_labels = convert_to_spin_op_labels(self._vibrational_data,
                                                         self._num_modes, self.num_modals)
        self._register_length = sum(self._num_modals) if isinstance(self._num_modals, list) \
            else self._num_modals * self._num_modes

        super().__init__(self._spin_op_labels, spin, self._register_length)

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
