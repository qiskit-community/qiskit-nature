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
"""A Vibrational Spin operator."""

from fractions import Fraction
from typing import List, Tuple, Union
from qiskit_nature.operators.second_quantization.vibrational_spin_op_utils \
    .vibr_to_spin_op_label_converter import _convert_to_spin_op_labels
from qiskit_nature.operators.second_quantization.vibrational_spin_op_utils \
    .vibrational_labels_validator import _validate_vibrational_labels
from .. import SpinOp


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
    :class:`VibrationalSpinOp` accepts the notation that encodes raising (+) and lowering (-)
    operators together with indices of modes and modals that they act on, e.g. "+_{mode_index}*{
    modal_index}". Each modal can be excited at most once.
    **Initialization**
    The :class:`VibrationalSpinOp` can be initialized by the list of tuples that each contains a
    string with a label as explained above and a corresponding coefficient. This argument must be
    accompanied by the number of modes and modals, and possibly, the value of a spin.
    **Algebra**
    :class:`VibrationalSpinOp` supports the following basic arithmetic operations: addition,
    subtraction,
    scalar multiplication, and dagger(adjoint).
    """

    def __init__(self, data: List[Tuple[str, complex]], num_modes: int,
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
                        that describes the number of them; the total number of modals defines a
                        `register_length`
            spin: positive half-integer (integer or half-odd-integer) that represents spin.
        Raises:
            ValueError: invalid data is given.
            QiskitNatureError: invalid spin value.
        """
        _validate_vibrational_labels(data, num_modes, num_modals)

        self._vibrational_data = data
        self._num_modals = num_modals
        self._num_modes = num_modes

        self._spin_op_labels = _convert_to_spin_op_labels(self._vibrational_data,
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
