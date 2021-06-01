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
from functools import reduce
from typing import List, Optional, Tuple, Union, cast

from qiskit_nature.operators.second_quantization.legacy_fermionic_op import _LegacyFermionicOp
from qiskit_nature.operators.second_quantization.second_quantized_op import SecondQuantizedOp


class FermionicOp(SecondQuantizedOp):
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
    The label mode is automatically detected by the presence of underscore `_`.

    1. Dense Label

    Dense labels are strings with allowed characters above.
    This is similar to Qiskit's string-based representation of qubit operators.
    For example,

    .. code-block:: python

        "+"
        "II++N-IE"

    are possible labels.

    2. Sparse Label

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
    _label_display_mode_is_dense: bool = True

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

        if all("_" not in label for label, _ in data):
            data = [
                (" ".join(f"{c}_{i}" for i, c in enumerate(label)), coeff) for label, coeff in data
            ]

        label_pattern = re.compile(r"^[\+\-NIE]_\d+$")
        invalid_labels = [
            label for label, _ in data if not all(label_pattern.match(c) for c in label.split())
        ]
        if invalid_labels:
            raise ValueError(f"Invalid labels for sparse labels are given: {invalid_labels}")

        self._register_length = (
            register_length
            if register_length
            else max(max((int(c[2:]) for c in label.split()), default=0) for label, _ in data) + 1
        )

        self._data = [(self._substituted_label(label), complex(coeff)) for label, coeff in data]

    @staticmethod
    def _substituted_label(label):
        re_number = re.compile(r"N_(\d+)")
        re_empty = re.compile(r"E_(\d+)")
        substituted_label = re_number.sub(r"+_\1 -_\1", re_empty.sub(r"-_\1 +_\1", label))
        return " ".join(filter(lambda x: x[0] != "I", substituted_label.split()))

    def __repr__(self) -> str:
        data = self.to_list()
        if len(self) == 1:
            if data[0][1] == 1:
                return f"FermionicOp('{data[0][0]}', register_length={self.register_length})"
            return f"FermionicOp({data[0]}, register_length={self.register_length})"
        return f"FermionicOp({data}, register_length={self.register_length})"  # TODO truncate

    def __str__(self) -> str:
        """Sets the representation of `self` in the console."""
        if len(self) == 1:
            label, coeff = self.to_list()[0]
            return f"{label} * {coeff}"
        return "  " + "\n+ ".join([f"{label} * {coeff}" for label, coeff in self.to_list()])

    def __len__(self):
        return len(self._data)

    @property
    def register_length(self) -> int:
        """Gets the register length."""
        return self._register_length

    def mul(self, other: complex) -> "FermionicOp":
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'FermionicOp' and '{type(other).__name__}'"
            )
        return FermionicOp(
            [(label, coeff * other) for label, coeff in self._data],
            register_length=self.register_length,
        )

    def compose(self, other: "FermionicOp") -> "FermionicOp":
        if not isinstance(other, FermionicOp):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'FermionicOp' and '{type(other).__name__}'"
            )

        new_data = list(
            filter(
                lambda x: x[1] != 0,
                (
                    (" ".join(filter(None, (label1, label2))), cf1 * cf2)
                    for label2, cf2 in other._data
                    for label1, cf1 in self._data
                ),
            )
        )
        register_length = max(self.register_length, other.register_length)
        if not new_data:
            return FermionicOp(("", 0), register_length)
        return FermionicOp._from_legacy(FermionicOp(new_data, register_length)._to_legacy())

    def add(self, other: "FermionicOp") -> "FermionicOp":
        if not isinstance(other, FermionicOp):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'FermionicOp' and '{type(other).__name__}'"
            )

        return FermionicOp(
            self._data + other._data,
            max(self.register_length, other.register_length),
        )

    def to_list(self) -> List[Tuple[str, complex]]:
        """Returns the operators internal contents in list-format.

        Returns:
            A list of tuples consisting of the dense label and corresponding coefficient.
        """
        if FermionicOp._label_display_mode_is_dense:
            return self._to_legacy().to_list()
        return self._data.copy()

    def adjoint(self) -> "FermionicOp":
        data = []
        for label, coeff in self._data:
            conjugated_coeff = coeff.conjugate()
            adjoint_label = " ".join(
                "+" + c[1:] if c[0] == "-" else "-" + c[1:] for c in reversed(label.split())
            )
            data.append((adjoint_label, conjugated_coeff))

        return FermionicOp(data, register_length=self.register_length)

    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None) -> "FermionicOp":
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        op = self._to_legacy().reduce(atol, rtol)
        return FermionicOp._from_legacy(op)

    @classmethod
    def set_label_display_mode(cls, mode: str):
        """Set the display mode of labels.

        Args:
            mode: display mode of labels. "sparse" or "dense" is available.

        Raises:
            ValueError: invalid mode is given
        """
        mode_lower = mode.lower()
        if mode_lower == "dense":
            cls._label_display_mode_is_dense = True
        elif mode_lower == "sparse":
            cls._label_display_mode_is_dense = False
        else:
            raise ValueError(f"Invalid `mode` {mode} is given. `mode` must be dense or sparse.")

    @classmethod
    def _from_legacy(cls, op: _LegacyFermionicOp):
        return cls(
            [(cls._to_sparse_label(label), coeff) for label, coeff in op.to_list()],
            register_length=op.register_length,
        )

    def _to_legacy(self) -> _LegacyFermionicOp:
        op = sum(
            (
                reduce(
                    lambda a, b: a.compose(b),
                    (_LegacyFermionicOp(c, self.register_length) for c in label.split()),
                )
                if label != ""
                else _LegacyFermionicOp("", self.register_length)
            )
            * coeff
            for label, coeff in self._data
        )
        op = cast(_LegacyFermionicOp, op)
        non_zero_list = [elem for elem in op.to_list() if elem[1] != 0]
        if not non_zero_list:
            return _LegacyFermionicOp(("I" * self.register_length, 0))
        return _LegacyFermionicOp(non_zero_list)

    @staticmethod
    def _to_sparse_label(label):
        label_transformation = {
            "I": "",
            "N": "+_{i} -_{i}",
            "E": "-_{i} +_{i}",
            "+": "+_{i}",
            "-": "-_{i}",
        }
        return " ".join(
            filter(None, (label_transformation[c].format(i=i) for i, c in enumerate(label)))
        )
