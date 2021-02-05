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

from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np

from qiskit_nature import QiskitNatureError

from .particle_op import ParticleOp


class FermionicOp(ParticleOp):
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

    `FermionicOp` supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, operator multiplication, and dagger(adjoint).
    For example,

    .. jupyter-execute::

      from qiskit_nature.operators import FermionicOp

      print("Addition")
      print(0.5 * FermionicOp("I+") + FermionicOp("+I"))
      print("Sum")
      print(0.25 * sum(FermionicOp(label) for label in ['NIII', 'INII', 'IINI', 'IIIN']))
      print("Operator multiplication")
      print(FermionicOp("+-") @ FermionicOp("E+"))
      print("Dagger")
      print(FermionicOp("+").dagger)

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
    ):
        """Initialize the FermionicOp.

        Args:
            data: Input data for FermionicOp. The allowed data is label str,
                  tuple (label, coeff), or list [(label, coeff)].

        Raises:
            QiskitNatureError: given data is invalid.
        """
        if not isinstance(data, (tuple, list, str)):
            raise QiskitNatureError("Invalid input data for FermionicOp.")

        if isinstance(data, tuple):
            if isinstance(data[0], str) and isinstance(data[1], (int, float, complex)):
                label = data[0]
                if not self._validate_label(label):
                    raise QiskitNatureError(
                        "Label must be a string consisting only of "
                        f"['I','+','-','N','E'] not: {label}"
                    )
                self._register_length = len(label)
                self._labels = [label]
                self._coeffs = [data[1]]
            else:
                raise QiskitNatureError(
                    "Data tuple must be (str, number), "
                    f"but ({type(data[0])}, {type(data[1])}) is given."
                )

        elif isinstance(data, str):
            label = data
            if not self._validate_label(label):
                raise QiskitNatureError(
                    "Label must be a string consisting only of "
                    f"['I','+','-','N','E'] not: {label}"
                )
            self._register_length = len(label)
            self._labels = [label]
            self._coeffs = [1]

        elif isinstance(data, list):
            if all(
                    isinstance(datum[0], str)
                    and isinstance(datum[1], (int, float, complex))
                    for datum in data
            ):
                self._register_length = len(data[0][0])
                self._labels, self._coeffs = zip(*data)  # type: ignore
                if not all(self._validate_label(label) for label in self._labels):
                    raise QiskitNatureError("Invalid labels are given.")
            else:
                raise QiskitNatureError("Data list must be [(str, number)].")

    def __repr__(self) -> str:
        if len(self) == 1:
            if self._coeffs == 1:
                return f"FermionicOp('{self._labels[0]}')"
            else:
                return f"FermionicOp(('{self._labels[0]}', {self._coeffs[0]}))"
        return f"FermionicOp({self.to_list()})"  # TODO truncate

    def __str__(self) -> str:
        """Sets the representation of `self` in the console."""

        # 1. Treat the case of the zero-operator:
        if len(self) == 0:
            return "Empty operator ({})".format(self.register_length)

        # 2. Treat the general case:
        if len(self) == 1:
            label, coeff = self.to_list()[0]
            return f"{label} * {coeff}"
        return "  " + "\n+ ".join(
            [f"{label} * {coeff}" for label, coeff in self.to_list()]
        )

    def mul(self, other: complex) -> "FermionicOp":
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'FermionicOp' and '{type(other).__name__}'"
            )
        return FermionicOp(
            list(zip(self._labels, [coeff * other for coeff in self._coeffs]))
        )

    def compose(self, other: "FermionicOp") -> "FermionicOp":
        if isinstance(other, FermionicOp):
            # Initialize new operator_list for the returned Fermionic operator
            new_data = []

            # Compute the product (Fermionic type operators consist of a sum of
            # FermionicOperator): F1 * F2 = (B1 + B2 + ...) * (C1 + C2 + ...) where Bi and Ci
            # are FermionicOperators
            for label1, cf1 in self.to_list():
                for label2, cf2 in other.to_list():
                    new_label, new_coeff = self._single_mul(label1, label2)
                    if new_coeff == 0:
                        continue
                    new_data.append((new_label, cf1 * cf2 * new_coeff))

            if not new_data:
                return FermionicOp(("I" * self._register_length, 0))

            return FermionicOp(new_data)

        raise TypeError(
            "Unsupported operand type(s) for *: 'FermionicOp' and "
            "'{}'".format(type(other).__name__)
        )

    @staticmethod
    def _single_mul(label1: str, label2: str) -> Tuple[str, complex]:
        assert len(label1) == len(
            label2
        ), "Operators act on Fermion Registers of different length"

        new_label = ["I"] * len(label1)
        new_coeff = 1

        # Map the products of two operators on a single fermionic mode to their result.
        mapping: Dict[str, Union[str, int]] = {
            # 0                   - if product vanishes,
            # new label           - if product does not vanish
            "II": "I",
            "I+": "+",
            "I-": "-",
            "IN": "N",
            "IE": "E",
            "+I": "+",
            "++": 0,
            "+-": "N",
            "+N": 0,
            "+E": "+",
            "-I": "-",
            "-+": "E",
            "--": 0,
            "-N": "-",
            "-E": 0,
            "NI": "N",
            "N+": "+",
            "N-": 0,
            "NN": "N",
            "NE": 0,
            "EI": "E",
            "E+": 0,
            "E-": "-",
            "EN": 0,
            "EE": "E",
        }

        # count the number of `+` and `-` in the first label ahead of time
        count = label1.count("+") + label1.count("-")

        for i, char1, char2 in zip(range(len(label1)), label1, label2):
            # update the count as we progress
            if char1 in "+-":
                count -= 1

            # Check what happens to the symbol
            new_char = mapping[char1 + char2]
            if new_char == 0:
                return "I" * len(label1), 0
            new_label[i] = new_char

            # If char2 is one of `+` or `-` we pick up a phase when commuting it to the position
            # of char1. However, we only care about this if the number of permutations has odd
            # parity.
            if count % 2 and char2 in "+-":
                new_coeff *= -1

        return ''.join(new_label), new_coeff

    def add(self, other: "FermionicOp") -> "FermionicOp":
        if not isinstance(other, FermionicOp):
            raise TypeError(
                "Unsupported operand type(s) for +: 'FermionicOp' and "
                "'{}'".format(type(other).__name__)
            )

        # Check compatibility (i.e. operators act on same register length)
        if self.register_length != other.register_length:
            raise TypeError("Incompatible register lengths for '+'.")

        label1, coeffs1 = zip(*self.to_list())
        label2, coeffs2 = zip(*other.to_list())

        return FermionicOp(list(zip(label1 + label2, coeffs1 + coeffs2)))

    def to_list(self) -> List[Tuple[str, complex]]:
        """Getter for the operator_list of `self`"""
        return list(zip(self._labels, self._coeffs))

    @property
    def register_length(self) -> int:
        """Getter for the length of the fermionic register that the FermionicOp `self` acts
        on.
        """
        return self._register_length

    def adjoint(self) -> "FermionicOp":
        dagger_map = {"+": "-", "-": "+", "I": "I", "N": "N", "E": "E"}
        label_list = []
        coeff_list = []
        for label, coeff in zip(self._labels, self._coeffs):
            conjugated_coeff = coeff.conjugate()

            daggered_label = ""

            for i, char in enumerate(label):
                daggered_label += dagger_map[char]
                if char in ["+", "-"]:
                    permute_through = label[i + 1:]
                    conjugated_coeff *= (-1) ** (
                        permute_through.count("+") + permute_through.count("-")
                    )

            label_list.append(daggered_label)
            coeff_list.append(conjugated_coeff)

        return FermionicOp(list(zip(label_list, coeff_list)))

    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None) -> "FermionicOp":
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        label_list, indexes = np.unique(self._labels, return_inverse=True, axis=0)
        coeff_list = np.zeros(len(self._coeffs))
        for i, val in zip(indexes, self._coeffs):
            coeff_list[i] += val
        non_zero = [
            i
            for i, v in enumerate(coeff_list)
            if not np.isclose(v, 0, atol=atol, rtol=rtol)
        ]
        label_list = label_list[non_zero]
        coeff_list = coeff_list[non_zero]
        if not non_zero:
            return FermionicOp(("I" * self.register_length, 0))
        return FermionicOp(list(zip(label_list, coeff_list)))

    def __len__(self):
        return len(self._labels)

    @staticmethod
    def _validate_label(label: str) -> bool:
        return all(char in ["I", "+", "-", "N", "E"] for char in label)
