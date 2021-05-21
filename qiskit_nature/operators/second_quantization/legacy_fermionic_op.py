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

"""The Fermionic-particle Operator."""

import re
from typing import List, Optional, Tuple, Union

import numpy as np
from qiskit.utils.validation import validate_min, validate_range_exclusive_max

from qiskit_nature import QiskitNatureError
from .second_quantized_op import SecondQuantizedOp


class LegacyFermionicOp(SecondQuantizedOp):
    r"""
    N-mode Fermionic operator.

    **Label**

    Allowed characters for the label are `I`, `-`, `+`, `N`, and, `E`.

    .. list-table::
        :header-rows: 1

        * - Label
          - Mathematical Representation
          - Meaning
        * - `I`
          - :math:`I`
          - Identity operator
        * - `-`
          - :math:`c`
          - Annihilation operator
        * - `+`
          - :math:`c^\dagger`
          - Creation operator
        * - `N`
          - :math:`n = c^\dagger c`
          - Number operator
        * - `E`
          - :math:`I - n = c c^\dagger`
          - Hole number

    There are two types of label modes for this class.
    The label mode is automatically detected.

    1. Dense Label (default, `register_length = None`)

    Dense labels are strings with allowed characters above.
    This is similar to Qiskit's string-based representation of qubit operators.
    For example,

    .. code-block:: python

        "+"
        "II++N-IE"

    are possible labels.

    2. Sparse Label (`register_length` is passed)

    When the parameter `register_length` is passed to :meth:`~FermionicOp.__init__`,
    label is assumed to be a sparse label.
    A sparse label is a string consisting of a space-separated list of words.
    Each word must look like :code:`[+-INE]_<index>`, where the :code:`<index>`
    is a non-negative integer representing the index of the fermionic mode.
    For example,

    .. code-block:: python

        "+_0"
        "-_2"
        "+_0 -_1 +_4 +_10"

    are possible labels.
    The :code:`index` must be in ascending order, and it does not allow duplicated indices.
    Thus, :code:`"+_1 N_0"` and `+_0 -_0` are invalid labels.

    **Initialization**

    The FermionicOp can be initialized in several ways:

        `FermionicOp(label)`
          A label consists of the permitted characters listed above.

        `FermionicOp(tuple)`
          Valid tuples are of the form `(label, coeff)`. `coeff` can be either `int`, `float`,
          or `complex`.

        `FermionicOp(list)`
          The list must be a list of valid tuples as explained above.

    **Algebra**

    This class supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, operator multiplication, and dagger(adjoint).
    For example,

    Addition

    .. jupyter-execute::

      from qiskit_nature.operators.second_quantization import FermionicOp
      0.5 * FermionicOp("I+") + FermionicOp("+I")

    Sum

    .. jupyter-execute::

      0.25 * sum(FermionicOp(label) for label in ['NIII', 'INII', 'IINI', 'IIIN'])

    Operator multiplication

    .. jupyter-execute::

      print(FermionicOp("+-") @ FermionicOp("E+"))

    Dagger

    .. jupyter-execute::

      ~FermionicOp("+")

    In principle, you can also add :class:`FermionicOp` and integers, but the only valid case is the
    addition of `0 + FermionicOp`. This makes the `sum` operation from the example above possible
    and it is useful in the following scenario:

    .. code-block:: python

        fermion = 0
        for i in some_iterable:
            some processing
            fermion += FermionicOp(somedata)

    """

    def __init__(
        self,
        data: Union[str, Tuple[str, complex], List[Tuple[str, complex]]],
        register_length: Optional[int] = None,
    ):
        """
        Args:
            data: Input data for FermionicOp. The allowed data is label str,
                  tuple (label, coeff), or list [(label, coeff)].
            register_length: positive integer that represents the length of registers.

        Raises:
            ValueError: given data is invalid value.
            TypeError: given data has invalid type.
        """
        self._register_length: int
        self._coeffs: np.ndarray
        self._labels: List[str]

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

        labels, coeffs = zip(*data)
        self._coeffs = np.array(coeffs, np.complex128)

        if register_length is None:  # Dense label
            self._register_length = len(labels[0])
            if not all(len(label) == self._register_length for label in labels):
                raise ValueError("Lengths of strings of label are different.")
            label_pattern = re.compile(r"^[I\+\-NE]+$")
            invalid_labels = [label for label in labels if not label_pattern.match(label)]
            if invalid_labels:
                raise ValueError(f"Invalid labels for dense labels are given: {invalid_labels}")
            self._labels = list(labels)
        else:  # Sparse label
            validate_min("register_length", register_length, 1)
            self._register_length = register_length
            label_pattern = re.compile(r"^[I\+\-NE]_\d+$")
            invalid_labels = [
                label
                for label in labels
                if not all(label_pattern.match(lb) for lb in label.split())
            ]
            if invalid_labels:
                raise ValueError(f"Invalid labels for sparse labels are given: {invalid_labels}")
            list_label = [["I"] * self._register_length for _ in labels]
            for term, label in enumerate(labels):
                prev_index: Optional[int] = None
                for split_label in label.split():
                    op_label, index_str = split_label.split("_", 1)
                    index = int(index_str)
                    validate_range_exclusive_max("index", index, 0, self._register_length)
                    if prev_index is not None and prev_index > index:
                        raise ValueError("Indices of labels must be in ascending order.")
                    if list_label[term][index] != "I":
                        raise ValueError(f"Duplicate index {index} is given.")
                    list_label[term][index] = op_label
                    prev_index = index

            self._labels = ["".join(lb) for lb in list_label]

    def __repr__(self) -> str:
        if len(self) == 1:
            if self._coeffs[0] == 1:
                return f"LegacyFermionicOp('{self._labels[0]}')"
            return f"LegacyFermionicOp({self.to_list()[0]})"
        return f"LegacyFermionicOp({self.to_list()})"  # TODO truncate

    def __str__(self) -> str:
        """Sets the representation of `self` in the console."""
        if len(self) == 1:
            label, coeff = self.to_list()[0]
            return f"{label} * {coeff}"
        return "  " + "\n+ ".join([f"{label} * {coeff}" for label, coeff in self.to_list()])

    def __len__(self):
        return len(self._labels)

    @property
    def register_length(self) -> int:
        """Gets the register length."""
        return self._register_length

    def mul(self, other: complex) -> "LegacyFermionicOp":
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'LegacyFermionicOp' and '{type(other).__name__}'"
            )
        return LegacyFermionicOp(list(zip(self._labels, (other * self._coeffs).tolist())))

    def compose(self, other: "LegacyFermionicOp") -> "LegacyFermionicOp":
        if isinstance(other, LegacyFermionicOp):
            # Initialize new operator_list for the returned Fermionic operator
            new_data = []

            # Compute the product (Fermionic type operators consist of a sum of
            # FermionicOperator): F1 * F2 = (B1 + B2 + ...) * (C1 + C2 + ...) where Bi and Ci
            # are FermionicOperators
            for label1, cf1 in self.to_list():
                for label2, cf2 in other.to_list():
                    new_label, sign = self._single_mul(label1, label2)
                    if sign == 0:
                        continue
                    new_data.append((new_label, cf1 * cf2 * sign))

            if not new_data:
                return LegacyFermionicOp(("I" * self._register_length, 0))

            return LegacyFermionicOp(new_data)

        raise TypeError(
            f"Unsupported operand type(s) for *: 'FermionicOp' and '{type(other).__name__}'"
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

            new_char = cls._MAPPING[pair]
            if new_char == "0":
                # if the new symbol is a zero-op, return early
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

    def add(self, other: "LegacyFermionicOp") -> "LegacyFermionicOp":
        if not isinstance(other, LegacyFermionicOp):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'LegacyFermionicOp' and '{type(other).__name__}'"
            )

        # Check compatibility (i.e. operators act on same register length)
        if self.register_length != other.register_length:
            raise TypeError("Incompatible register lengths for '+'.")

        return LegacyFermionicOp(
            list(
                zip(
                    self._labels + other._labels,
                    np.hstack((self._coeffs, other._coeffs)).tolist(),
                )
            )
        )

    def to_list(self) -> List[Tuple[str, complex]]:
        """Returns the operators internal contents in list-format.

        Returns:
            A list of tuples consisting of the dense label and corresponding coefficient.
        """
        return list(zip(self._labels, self._coeffs.tolist()))

    def adjoint(self) -> "LegacyFermionicOp":
        dagger_map = {"+": "-", "-": "+", "I": "I", "N": "N", "E": "E"}
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

        return LegacyFermionicOp(list(zip(label_list, np.array(coeff_list, dtype=np.complex128))))

    def reduce(
        self, atol: Optional[float] = None, rtol: Optional[float] = None
    ) -> "LegacyFermionicOp":
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
            return LegacyFermionicOp(("I" * self.register_length, 0))
        return LegacyFermionicOp(list(zip(label_list[non_zero].tolist(), coeff_list[non_zero])))
