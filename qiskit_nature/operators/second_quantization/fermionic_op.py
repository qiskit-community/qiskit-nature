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
import warnings
from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit_nature.operators.second_quantization.second_quantized_op import SecondQuantizedOp

_ZERO_LABELS = {
    ("+", "+"),
    ("+", "N"),
    ("-", "-"),
    ("-", "E"),
    ("N", "E"),
    ("E", "+"),
    ("N", "-"),
    ("E", "N"),
}
_MAPPING = {
    ("I", "I"): "I",
    ("I", "+"): "+",
    ("I", "-"): "-",
    ("I", "N"): "N",
    ("I", "E"): "E",
    ("+", "I"): "+",
    ("+", "-"): "N",
    ("+", "E"): "+",
    ("-", "I"): "-",
    ("-", "+"): "E",
    ("-", "N"): "-",
    ("N", "I"): "N",
    ("N", "+"): "+",
    ("N", "N"): "N",
    ("E", "I"): "E",
    ("E", "-"): "-",
    ("E", "E"): "E",
}


@dataclass(frozen=True)
class _FermionLabelPrimitive:
    """Represent Label for Fermion +_{n} or -_{n}"""

    is_creation: bool  # if True creation operator, otherwise annihilation operator
    index: int

    def __str__(self):
        if self.is_creation:
            return f"+_{self.index}"
        return f"-_{self.index}"

    def adjoint(self) -> "_FermionLabelPrimitive":
        """Calculate adjoint

        Returns:
            The adjoint label
        """
        return _FermionLabelPrimitive(not self.is_creation, self.index)


_FermionLabel = List[_FermionLabelPrimitive]


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

    **Output of str and repr**

    By default, the output of str and repr is truncated.
    You can change the number of characters with `set_truncation`.
    If you pass 0 to `set_truncation`, truncation is disabled and the full output will be printed.

    Example:

    .. jupyter-execute::

      from qiskit_nature.operators.second_quantization import FermionicOp

      print("truncated str output")
      print(sum(FermionicOp("I", display_format="sparse") for _ in range(25)))

      FermionicOp.set_truncation(0)
      print("not truncated str output")
      print(sum(FermionicOp("I", display_format="sparse") for _ in range(25)))


    **Algebra**

    This class supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, operator multiplication, and adjoint.
    For example,

    Addition

    .. jupyter-execute::

      0.5 * FermionicOp("I+", display_format="dense") + FermionicOp("+I", display_format="dense")

    Sum

    .. jupyter-execute::

      0.25 * sum(FermionicOp(label, display_format="sparse") for label in ['+_0', '-_1', 'N_2'])

    Operator multiplication

    .. jupyter-execute::

      print(FermionicOp("+-", display_format="dense") @ FermionicOp("E+", display_format="dense"))

    Dagger

    .. jupyter-execute::

      ~FermionicOp("+", display_format="dense")

    In principle, you can also add :class:`FermionicOp` and integers, but the only valid case is the
    addition of `0 + FermionicOp`. This makes the `sum` operation from the example above possible
    and it is useful in the following scenario:

    .. code-block:: python

        fermion = 0
        for i in some_iterable:
            some processing
            fermion += FermionicOp(somedata)

    """
    # Warn only once
    _display_format_warn = True

    _truncate = 200

    def __init__(
        self,
        data: Union[
            str,
            Tuple[str, complex],
            List[Tuple[str, complex]],
            List[Tuple[str, float]],
            List[Tuple[_FermionLabel, complex]],
        ],
        register_length: Optional[int] = None,
        display_format: Optional[str] = None,
    ):
        """
        Args:
            data: Input data for FermionicOp. The allowed data is label str,
                  tuple (label, coeff), or list [(label, coeff)].
            register_length: positive integer that represents the length of registers.
            display_format: If sparse, the label is represented sparsely during output.
                            if dense, the label is represented densely during output. (default: dense)

        Raises:
            ValueError: given data is invalid value.
            TypeError: given data has invalid type.
        """
        if display_format is None:
            display_format = "dense"
            if FermionicOp._display_format_warn:
                FermionicOp._display_format_warn = False
                warnings.warn(
                    "The default value for `display_format` will be changed from 'dense' "
                    "to 'sparse' in version 0.3.0. Once that happens, you must specify "
                    "display_format='dense' directly.",
                    stacklevel=2,
                )

        self.display_format = display_format

        self._data: List[Tuple[_FermionLabel, complex]]

        if isinstance(data, list) and isinstance(data[0][0], list) and register_length is not None:
            self._data = data  # type: ignore
            self._register_length = register_length
        else:
            if not isinstance(data, (tuple, list, str)):
                raise TypeError(f"Type of data must be str, tuple, or list, not {type(data)}.")

            if isinstance(data, str):
                data = [(data, complex(1))]

            elif isinstance(data, tuple):
                if not isinstance(data[0], str) or not isinstance(data[1], (int, float, complex)):
                    raise TypeError(
                        f"Data tuple must be (str, number), not ({type(data[0])}, {type(data[1])})."
                    )
                data = [data]

            else:
                if not isinstance(data[0][0], str) or not isinstance(
                    data[0][1], (int, float, complex)
                ):
                    raise TypeError("Data list must be [(str, number)].")

            if all("_" not in label for label, _ in data):
                self._data = [
                    (
                        self._substituted_label([(c, int(i)) for i, c in enumerate(label)]),
                        complex(coeff),  # type: ignore
                    )
                    for label, coeff in data
                ]
            else:
                self._data = [
                    (
                        self._substituted_label(
                            [(c[0], int(c[2:])) for c in label.split()]  # type: ignore
                        ),
                        complex(coeff),  # type: ignore
                    )
                    for label, coeff in data
                ]

            if register_length is not None:
                self._register_length = register_length

    def _substituted_label(self, label):
        max_index = 0
        new_label = []
        for c, index in label:
            max_index = max(max_index, index)
            if c == "+":
                new_label.append(_FermionLabelPrimitive(True, index))
            elif c == "-":
                new_label.append(_FermionLabelPrimitive(False, index))
            elif c == "N":
                new_label.append(_FermionLabelPrimitive(True, index))
                new_label.append(_FermionLabelPrimitive(False, index))
            elif c == "E":
                new_label.append(_FermionLabelPrimitive(False, index))
                new_label.append(_FermionLabelPrimitive(True, index))
            elif c == "I":
                continue
            else:
                raise ValueError(f"Invalid label {c}_{index} is given.")

        self._register_length = max_index + 1
        return new_label

    def __repr__(self) -> str:
        data = self.to_list()
        if len(self) == 1:
            if data[0][1] == 1:
                data_str = f"'{data[0][0]}'"
            data_str = f"'{data[0]}'"
        data_str = f"{data}"

        if FermionicOp._truncate and len(data_str) > FermionicOp._truncate:
            data_str = data_str[0 : FermionicOp._truncate - 5] + "..." + data_str[-2:]
        return (
            "FermionicOp("
            f"{data_str}, "
            f"register_length={self.register_length}, "
            f"display_format='{self.display_format}'"
            ")"
        )

    @classmethod
    def set_truncation(cls, val: int) -> None:
        """Set the max number of characters to display before truncation.
        Args:
            val: the number of characters.

        .. note::
            Truncation will be disabled if the truncation value is set to 0.
        """
        cls._truncate = int(val)

    def __str__(self) -> str:
        """Sets the representation of `self` in the console."""

        if len(self) == 1:
            label, coeff = self.to_list()[0]
            return f"{coeff} * ({label})"
        pre = (
            "Fermionic Operator\n"
            f"register length={self.register_length}, number terms={len(self)}\n"
        )
        ret = "  " + "\n+ ".join(
            [f"{coeff} * ( {label} )" if label else f"{coeff}" for label, coeff in self.to_list()]
        )
        if FermionicOp._truncate and len(ret) > FermionicOp._truncate:
            ret = ret[0 : FermionicOp._truncate - 4] + " ..."
        return pre + ret

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
            display_format=self.display_format,
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
                    (label1 + label2, cf1 * cf2)
                    for label2, cf2 in other._data
                    for label1, cf1 in self._data
                ),
            )
        )
        register_length = max(self.register_length, other.register_length)
        display_format = (
            "sparse"
            if self.display_format == "sparse" or other.display_format == "sparse"
            else "dense"
        )
        if not new_data:
            return FermionicOp(("", 0), register_length, display_format)
        return FermionicOp(new_data, register_length, display_format)

    def add(self, other: "FermionicOp") -> "FermionicOp":
        if not isinstance(other, FermionicOp):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'FermionicOp' and '{type(other).__name__}'"
            )

        return FermionicOp(
            self._data + other._data,
            max(self.register_length, other.register_length),
            self.display_format or other.display_format,
        )

    # pylint: disable=arguments-differ
    def to_list(
        self,
        display_format: Optional[str] = None,
    ) -> List[Tuple[str, complex]]:  # type: ignore
        """Returns the operators internal contents in list-format.

        Args:
            display_format: when specified this will overwrite ``self.display_format``. Can
                be either 'dense' or 'sparse'. See the class documentation for more details.

        Returns:
            A list of tuples consisting of the dense label and corresponding coefficient.

        Raises:
            ValueError: if the given format is invalid.
        """
        if display_format is not None:
            display_format = display_format.lower()
            if display_format not in {"sparse", "dense"}:
                raise ValueError(
                    f"Invalid `display_format` {display_format} is given."
                    "`display_format` must be 'dense' or 'sparse'."
                )
        else:
            display_format = self.display_format
        if display_format == "sparse":
            return [
                (" ".join(str(label) for label in label_data), coeff)
                for label_data, coeff in self._data
            ]
        return self._to_dense_label_data()

    def adjoint(self) -> "FermionicOp":
        data = []
        for label, coeff in self._data:
            conjugated_coeff = coeff.conjugate()
            adjoint_label = [fer_label.adjoint() for fer_label in reversed(label)]
            data.append((adjoint_label, conjugated_coeff))

        return FermionicOp(
            data, register_length=self.register_length, display_format=self.display_format
        )

    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None) -> "FermionicOp":
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        labels, coeffs = zip(*self.to_normal_order()._to_dense_label_data())
        label_list, indices = np.unique(labels, return_inverse=True, axis=0)
        coeff_list = np.zeros(len(coeffs), dtype=np.complex128)
        for i, val in zip(indices, coeffs):
            coeff_list[i] += val
        non_zero = [
            i for i, v in enumerate(coeff_list) if not np.isclose(v, 0, atol=atol, rtol=rtol)
        ]
        if not non_zero:
            return FermionicOp(("", 0), self.register_length)
        return FermionicOp(list(zip(label_list[non_zero].tolist(), coeff_list[non_zero])))

    @property
    def display_format(self):
        """Return the display format"""
        return self._display_format

    @display_format.setter
    def display_format(self, display_format: str):
        """Set the display format of labels.

        Args:
            display_format: display format for labels. "sparse" or "dense" is available.

        Raises:
            ValueError: invalid mode is given
        """
        display_format = display_format.lower()
        if display_format not in {"sparse", "dense"}:
            raise ValueError(
                f"Invalid `display_format` {display_format} is given."
                "`display_format` must be 'dense' or 'sparse'."
            )
        self._display_format = display_format

    def _to_dense_label_data(self) -> List[Tuple[str, complex]]:
        dense_label_data = []
        for label, coeff in self._data:
            label_list = ["I"] * self.register_length
            for fer_label in label:
                char = "+" if fer_label.is_creation else "-"
                index = fer_label.index
                if (label_list[index], char) in _ZERO_LABELS:
                    break
                label_list[index] = _MAPPING[(label_list[index], char)]
                if index != self.register_length and char in {"+", "-"}:
                    exchange_label = label_list[index + 1 :]
                    num_exchange = exchange_label.count("+") + exchange_label.count("-")
                    coeff *= -1 if num_exchange % 2 else 1
            else:
                dense_label_data.append(("".join(label_list), coeff))
        if not dense_label_data:
            return [("I" * self.register_length, 0j)]
        return dense_label_data

    def to_normal_order(self) -> "FermionicOp":
        """Convert to the equivalent operator with normal order.
        The returned operator is a sparse label mode.

        .. note::

            This method implements the transformation of an operator to the normal ordered operator.
            The transformation is calculated by considering all commutation relations between the
            operators. For example, for the case :math:`\\colon c_0 c_0^\\dagger\\colon`
            where :math:`c_0` is an annihilation operator,
            this method returns :math:`1 - c_0^\\dagger c_0` due to commutation relations.
            See the reference: https://en.wikipedia.org/wiki/Normal_order#Multiple_fermions.

        """
        temp_display_label = self.display_format
        self.display_format = "dense"
        ret = 0

        for label, coeff in self.to_list():
            splits = label.split("E")

            for inter_ops in product("IN", repeat=len(splits) - 1):
                label = splits[0]
                label += "".join(link + next_base for link, next_base in zip(inter_ops, splits[1:]))

                pluses = [it.start() for it in re.finditer(r"\+|N", label)]
                minuses = [it.start() for it in re.finditer(r"-|N", label)]

                count = sum(1 for plus in pluses for minus in minuses if plus > minus)
                sign_swap = (-1) ** count
                sign_n = (-1) ** inter_ops.count("N")
                new_coeff = coeff * sign_n * sign_swap

                ret += new_coeff * FermionicOp(
                    " ".join([f"+_{i}" for i in pluses] + [f"-_{i}" for i in minuses]),
                    self.register_length,
                    "sparse",
                )

        self.display_format = temp_display_label

        if isinstance(ret, FermionicOp):
            return ret
        return FermionicOp(("", 0), self.register_length, "sparse")

    @classmethod
    def zero(cls, register_length: int) -> "FermionicOp":
        """Constructs a zero-operator.

        Args:
            register_length: the length of the operator.

        Returns:
            The zero-operator of the given length.
        """
        return FermionicOp(("I_0", 0.0), register_length=register_length)

    @classmethod
    def one(cls, register_length: int) -> "FermionicOp":
        """Constructs a unity-operator.

        Args:
            register_length: the length of the operator.

        Returns:
            The unity-operator of the given length.
        """
        return FermionicOp(("I_0", 1.0), register_length=register_length)
