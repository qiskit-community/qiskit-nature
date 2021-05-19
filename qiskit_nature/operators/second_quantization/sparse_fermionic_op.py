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

"""The Sparse Fermionic-particle Operator."""

import re
from functools import reduce
from typing import List, Optional, Tuple, Union, cast

from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.operators.second_quantization.second_quantized_op import SecondQuantizedOp


class SparseFermionicOp(SecondQuantizedOp):
    r"""
    Sparse Fermionic operator.
    """

    def __init__(
        self,
        data: Union[str, Tuple[str, complex], List[Tuple[str, complex]]],
    ):
        """
        Args:
            data: Input data for SparseFermionicOp. The allowed data is label str,
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

        label_pattern = re.compile(r"^[\+\-]_\d+$")
        invalid_labels = [
            label for label, _ in data if not all(label_pattern.match(c) for c in label.split())
        ]
        if invalid_labels:
            raise ValueError(f"Invalid labels for sparse labels are given: {invalid_labels}")
        self._data = [(label, complex(coeff)) for label, coeff in data]

    def __repr__(self) -> str:
        if len(self) == 1:
            if self._data[0][1] == 1:
                return f"SparseFermionicOp('{self._data[0][0]}')"
            return f"SparseFermionicOp({self._data[0]})"
        return f"SparseFermionicOp({self._data})"  # TODO truncate

    def __str__(self) -> str:
        """Sets the representation of `self` in the console."""
        if len(self) == 1:
            label, coeff = self._data[0]
            return f"{label} * {coeff}"
        return "  " + "\n+ ".join([f"{label} * {coeff}" for label, coeff in self._data])

    def __len__(self):
        return len(self._data)

    @property
    def register_length(self) -> int:
        """Gets the register length."""
        return (
            max(max((int(c[2:]) for c in label.split()), default=0) for label, _ in self._data) + 1
        )

    def mul(self, other: complex) -> "SparseFermionicOp":
        if not isinstance(other, (int, float, complex)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SparseFermionicOp' and '{type(other).__name__}'"
            )
        return SparseFermionicOp([(label, coeff * other) for label, coeff in self._data])

    def compose(self, other: "SparseFermionicOp") -> "SparseFermionicOp":
        if not isinstance(other, SparseFermionicOp):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SparseFermionicOp' and '{type(other).__name__}'"
            )

        new_data = [
            (label1 + " " + label2, cf1 * cf2)
            for label2, cf2 in other._data
            for label1, cf1 in self._data
        ]
        return SparseFermionicOp(new_data)

    def add(self, other: "SparseFermionicOp") -> "SparseFermionicOp":
        if not isinstance(other, SparseFermionicOp):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'SparseFermionicOp' and '{type(other).__name__}'"
            )

        return SparseFermionicOp(self._data + other._data)

    def to_list(self) -> List[Tuple[str, complex]]:
        """Returns the operators internal contents in list-format.

        Returns:
            A list of tuples consisting of the dense label and corresponding coefficient.
        """
        return self._data

    def adjoint(self) -> "SparseFermionicOp":
        data = []
        for label, coeff in self._data:
            conjugated_coeff = coeff.conjugate()
            adjoint_label = " ".join(
                "+" + c[1:] if c[0] == "-" else "-" + c[1:] for c in reversed(label.split())
            )
            data.append((adjoint_label, conjugated_coeff))

        return SparseFermionicOp(data)

    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None) -> "FermionicOp":
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        op = sum(
            (
                reduce(
                    lambda a, b: a.compose(b),
                    (FermionicOp(c, self.register_length) for c in label.split()),
                )
                if label != ""
                else FermionicOp("", self.register_length)
            )
            * coeff
            for label, coeff in self._data
        )
        op = cast(FermionicOp, op)
        op = op.reduce(atol, rtol)
        return SparseFermionicOp._from_fermionic_op(op)

    @classmethod
    def _from_fermionic_op(cls, op: FermionicOp):
        return cls([(cls._to_sp_label(label), coeff) for label, coeff in op.to_list()])

    @staticmethod
    def _to_sp_label(label):
        label_transformation = {
            "I": "",
            "N": "+_{i} -_{i}",
            "E": "-_{i} +_{i}",
            "+": "+_{i}",
            "-": "-_{i}",
        }
        return " ".join([label_transformation[c].format(i=i) for i, c in enumerate(label)])
