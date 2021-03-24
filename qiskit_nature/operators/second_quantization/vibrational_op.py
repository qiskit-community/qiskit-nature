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
"""A Vibration operator."""

from functools import reduce
from typing import List, Tuple, Union, Optional

import re

import numpy as np

from qiskit_nature import QiskitNatureError

from .vibrational_op_utils.vibrational_labels_validator import _validate_vibrational_labels
from .vibrational_op_utils.vibrational_dense_label_converter import _convert_to_dense_labels

from .second_quantized_op import SecondQuantizedOp


class VibrationalOp(SecondQuantizedOp):
    """Vibration type operators.

    **Label**

    Allowed characters for primitives of labels are `+` and `-`.

    .. list-table::
        :header-rows: 1

        * - Label
          - Mathematical Representation
          - Meaning
        * - `+`
          - :math:`S_+`
          - Raising operator
        * - `-`
          - :math:`S_-`
          - Lowering operator

    :class:`VibrationalOp` accepts the notation that encodes raising (`+`) and lowering (`-`)
    operators together with indices of modes and modals that they act on, e.g.
    `+_{mode_index}*{modal_index}`. Each modal can be excited at most once.

    **Initialization**

    The :class:`VibrationalOp` can be initialized by the list of tuples that each contains a
    string with a label as explained above and a corresponding coefficient. This argument must be
    accompanied by the number of modes and modals, and possibly, the value of a spin.

    **Algebra**

    :class:`VibrationalOp` supports the following basic arithmetic operations: addition,
    subtraction, scalar multiplication, and dagger(adjoint).
    """

    def __init__(self, data: Union[str, Tuple[str, complex], List[Tuple[str, complex]]],
                 num_modes: int,
                 num_modals: Union[int, List[int]]):
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
            TypeError: given data has invalid type.
            ValueError: invalid labels.
        """
        if not isinstance(data, (tuple, list, str)):
            raise TypeError(f"Type of data must be str, tuple, or list, not {type(data)}.")

        if isinstance(data, tuple):
            if not isinstance(data[0], str) or not isinstance(data[1], (int, float, complex)):
                raise TypeError(
                    f"Data tuple must be (str, number), not ({type(data[0])}, {type(data[1])})."
                )
            data = [data]

        if isinstance(data, str):
            data = [(data, 1)]

        if not all(
            isinstance(label, str) and isinstance(coeff, (int, float, complex))
            for label, coeff in data
        ):
            raise TypeError("Data list must be [(str, number)].")

        self._coeffs: np.ndarray
        self._labels: List[str]

        if isinstance(num_modals, int):
            num_modals = [num_modals] * num_modes

        self._num_modes = num_modes
        self._num_modals = num_modals
        self._register_length = sum(self._num_modals)

        labels, coeffs = zip(*data)
        self._coeffs = np.array(coeffs, np.complex128)

        if not any('_' in label for label in labels):
            # Dense label
            if not all(len(label) == self._register_length for label in labels):
                raise ValueError("Lengths of strings of label are different.")
            label_pattern = re.compile(r"^[I\+\-NE]+$")
            invalid_labels = [label for label in labels if not label_pattern.match(label)]
            if invalid_labels:
                raise ValueError(f"Invalid labels for dense labels are given: {invalid_labels}")
            self._labels = list(labels)
        else:
            # Sparse label
            _validate_vibrational_labels(data, num_modes, num_modals)
            dense_labels = _convert_to_dense_labels(data, num_modes, num_modals)

            ops: List[VibrationalOp] = []
            for dense_label, coeff in dense_labels:
                new_op = reduce(lambda a, b: a @ b, (
                        VibrationalOp((label, 1), num_modes, num_modals) for label in dense_label
                ))
                # We ignore the type here because mypy only sees the complex coefficient
                ops.append(coeff * new_op)  # type: ignore

            op = reduce(lambda a, b: a + b, ops)

            self._labels = op._labels.copy()

    def __repr__(self) -> str:
        if len(self) == 1:
            if self._coeffs[0] == 1:
                return f"VibrationalOp('{self._labels[0]}')"
            return f"VibrationalOp({self.to_list()[0]})"
        return f"VibrationalOp({self.to_list()})"  # TODO truncate

    def __str__(self) -> str:
        """Sets the representation of `self` in the console."""
        if len(self) == 1:
            label, coeff = self.to_list()[0]
            return f"{label} * {coeff}"
        return "  " + "\n+ ".join([f"{label} * {coeff}" for label, coeff in self.to_list()])

    def __len__(self):
        return len(self._labels)

    @property
    def num_modes(self) -> int:
        """The number of modes.
        Returns:
            The number of modes
        """
        return self._num_modes

    @property
    def num_modals(self) -> List[int]:
        """The number of modals.
        Returns:
            The number of modals
        """
        return self._num_modals

    @property
    def register_length(self) -> int:
        """Getter for the length of the fermionic register that the VibrationalOp `self` acts
        on.
        """
        return self._register_length

    def mul(self, other: complex) -> "VibrationalOp":
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'VibrationalOp' and '{type(other).__name__}'"
            )
        return VibrationalOp(list(zip(self._labels, (other * self._coeffs).tolist())),
                             self._num_modes, self._num_modals)

    def add(self, other: "VibrationalOp") -> "VibrationalOp":
        if not isinstance(other, VibrationalOp):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'VibrationalOp' and '{type(other).__name__}'"
            )

        # Check compatibility
        if self.num_modes != other.num_modes or self.num_modals != other.num_modals:
            raise TypeError("Incompatible register lengths for '+'.")

        return VibrationalOp(
            list(
                zip(self._labels + other._labels, np.hstack((self._coeffs, other._coeffs)).tolist())
            ),
            self._num_modes,
            self._num_modals,
        )

    def to_list(self) -> List[Tuple[str, complex]]:
        """Getter for the operator_list of `self`"""
        return list(zip(self._labels, self._coeffs.tolist()))

    def adjoint(self) -> "VibrationalOp":
        dagger_map = {"+": "-", "-": "+", "I": "I"}
        label_list = []
        coeff_list = []
        for label, coeff in zip(self._labels, self._coeffs.tolist()):
            conjugated_coeff = coeff.conjugate()

            daggered_label = []
            count = label.count("+") + label.count("-")
            for char in label:
                daggered_label.append(dagger_map[char])
                if char in "+-":
                    count -= 1
                    if count % 2 == 1:
                        conjugated_coeff *= -1

            label_list.append("".join(daggered_label))
            coeff_list.append(conjugated_coeff)

        return VibrationalOp(list(zip(label_list, np.array(coeff_list, dtype=np.complex128))),
                             self._num_modes, self._num_modals)

    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None) -> "VibrationalOp":
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        label_list, indices = np.unique(self._labels, return_inverse=True, axis=0)
        coeff_list = np.zeros(len(self._coeffs), dtype=np.complex128)
        for i, val in zip(indices, self._coeffs):
            coeff_list[i] += val
        non_zero = [
            i for i, v in enumerate(coeff_list) if not np.isclose(v, 0, atol=atol, rtol=rtol)
        ]
        if not non_zero:
            return VibrationalOp(("I_0*0", 0), self._num_modes, self._num_modals)
        return VibrationalOp(list(zip(label_list[non_zero].tolist(), coeff_list[non_zero])),
                             self._num_modes, self._num_modals)

    def compose(self, other: "VibrationalOp") -> "VibrationalOp":
        if isinstance(other, VibrationalOp):
            # Initialize new operator_list for the returned Vibration operator
            new_data = []

            # Compute the product (Vibration type operators consist of a sum of
            # VibrationalOperator): F1 * F2 = (B1 + B2 + ...) * (C1 + C2 + ...) where Bi and Ci
            # are VibrationalOperators
            for label1, cf1 in self.to_list():
                for label2, cf2 in other.to_list():
                    new_label, sign = self._single_mul(label1, label2)
                    if sign == 0:
                        continue
                    # TODO: do we need to consider the sign in this application?
                    new_data.append((new_label, cf1 * cf2 * sign))

            if not new_data:
                return VibrationalOp(("I_0*0", 0), self._num_modes, self._num_modals)

            return VibrationalOp(new_data, self._num_modes, self._num_modals)

        raise TypeError(
            f"Unsupported operand type(s) for *: 'VibrationalOp' and '{type(other).__name__}'"
        )

    # Map the products of two operators on a single fermionic mode to their result.
    _MAPPING = {
        # 0                   - if product vanishes,
        # new label           - if product does not vanish
        ("I", "I"): "I",
        ("I", "+"): "+",
        ("I", "-"): "-",
        ("I", "N"): "N",
        ("I", "E"): "E",
        ("+", "I"): "+",
        ("+", "+"): "0",
        ("+", "-"): "N",
        ("+", "N"): "0",
        ("+", "E"): "+",
        ("-", "I"): "-",
        ("-", "+"): "E",
        ("-", "-"): "0",
        ("-", "N"): "-",
        ("-", "E"): "0",
        ("N", "I"): "N",
        ("N", "+"): "+",
        ("N", "-"): "0",
        ("N", "N"): "N",
        ("N", "E"): "0",
        ("E", "I"): "E",
        ("E", "+"): "0",
        ("E", "-"): "-",
        ("E", "N"): "0",
        ("E", "E"): "E",
    }

    @classmethod
    def _single_mul(cls, label1: str, label2: str) -> Tuple[str, complex]:
        if len(label1) != len(label2):
            raise QiskitNatureError("Operators act on Fermion Registers of different length")

        new_label = []
        sign = 1

        # count the number of `+` and `-` in the first label ahead of time
        count = label1.count("+") + label1.count("-")

        for pair in zip(label1, label2):
            # update the count as we progress
            char1, char2 = pair
            if char1 in "+-":
                count -= 1

            # Check what happens to the symbol
            new_char = cls._MAPPING[pair]
            if new_char == "0":
                return "I" * len(label1), 0
            new_label.append(new_char)
            # NOTE: we can ignore the type because the only scenario where an `int` occurs is caught
            # by the `if`-statement above.

            # If char2 is one of `+` or `-` we pick up a phase when commuting it to the position
            # of char1. However, we only care about this if the number of permutations has odd
            # parity.
            if count % 2 and char2 in "+-":
                sign *= -1

        return "".join(new_label), sign
