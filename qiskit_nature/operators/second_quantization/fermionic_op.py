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

from numbers import Number
from typing import cast, Dict, List, Tuple, Union

import numpy as np
from qiskit.opflow import PauliSumOp

from qiskit_nature import QiskitNatureError

from .particle_op import ParticleOp


class FermionicOp(ParticleOp):
    """
    Fermionic type operators

    The abstract fermionic registers are implemented in two subclasses, FermionicOperator and
    FermionicOp, inspired by the implementation of Pauli operators in qiskit. A
    FermionicOperator is the equivalent of a single Pauli string on a qubit register.
    The FermionicOp represents a sum of multiple FermionicOperators. They act on fermionic
    registers of a fixed length determined at the time of initialization.
    """

    def __init__(self, data, coeff=1, register_length=None):
        if not isinstance(data, (tuple, list, str)):
            raise QiskitNatureError("Invalid input data for FermionicOp.")

        if isinstance(data, tuple):
            if isinstance(data[0], str) and isinstance(data[1], Number):
                label = data[0]
                if not self._validate_label(label):
                    raise QiskitNatureError(
                        "Label must be a string consisting only of "
                        f"['I','+','-','N','E'] not: {label}"
                    )
                coeff = data[1]
                self._register_length = len(label)
                self._labels = [label]
                self._coeffs = [coeff]
            else:
                raise QiskitNatureError(
                    "Data tuple must be (str, Number), "
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
            self._coeffs = [coeff]

        elif isinstance(data, list):
            if not data and register_length:
                self._labels = ["I" * register_length]
                self._coeffs = [0]
            elif not data:
                raise QiskitNatureError(
                    "Empty data requires register_length parameter."
                )
            else:
                self._register_length = len(data[0][0])
                self._labels, self._coeffs = zip(*data)
                if not all(self._validate_label(label) for label in self._labels):
                    raise QiskitNatureError("Invalid labels are given.")

    def __repr__(self):
        if len(self) == 1:
            if self._coeffs == 1:
                return f"FermionicOp('{self._labels[0]}')"
            else:
                return f"FermionicOp('{self._labels[0]}', coeff={self._coeffs[0]})"
        return f"FermionicOp({self.to_list()})"

    def __str__(self):
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

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'FermionicSumOp' and '{type(other).__name__}'"
            )
        return FermionicOp(
            list(zip(self._labels, [coeff * other for coeff in self._coeffs]))
        )

    def __matmul__(self, other) -> "FermionicOp":
        """Overloads the multiplication operator `@` for self and other, where other is a
        number-type, a FermionicOperator or a FermionicOp.
        """
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
    def _single_mul(label1, label2) -> Tuple[str, complex]:
        assert len(label1) == len(
            label2
        ), "Operators act on Fermion Registers of different length"

        new_label = ""
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

        for i, char1, char2 in zip(range(len(label1)), label1, label2):
            # if char2 is one of `-`, `+` we pick up a phase when commuting it to the position
            # of char1
            if char2 in ["-", "+"]:
                # Construct the string through which we have to commute
                permuting_through = label1[i + 1:]
                # Count the number of times we pick up a minus sign when commuting
                ncommutations = permuting_through.count("+") + permuting_through.count(
                    "-"
                )
                new_coeff *= (-1) ** ncommutations

            # Check what happens to the symbol
            new_char = mapping[char1 + char2]
            if new_char == 0:
                return "I" * len(label1), 0
            new_label += cast(str, new_char)

        return new_label, new_coeff

    def __add__(self, other):
        """Returns a `FermionicOp` representing the sum of the given base fermionic
        operators.
        """
        if not isinstance(other, FermionicOp):
            raise TypeError(
                "Unsupported operand type(s) for +: 'FermionicOp' and "
                "'{}'".format(type(other).__name__)
            )

        # Check compatibility (i.e. operators act on same register length)
        assert self._is_compatible(other), "Incompatible register lengths for '+'. "

        label1, coeffs1 = zip(*self.to_list())
        label2, coeffs2 = zip(*other.to_list())

        return FermionicOp(list(zip(label1 + label2, coeffs1 + coeffs2))).reduce()

    def to_list(self) -> List[Tuple[str, complex]]:
        """Getter for the operator_list of `self`"""
        return list(zip(self._labels, self._coeffs))

    @property
    def register_length(self):
        """Getter for the length of the fermionic register that the FermionicOp `self` acts
        on.
        """
        return self._register_length

    def dagger(self):
        """Returns the complex conjugate transpose (dagger) of `self`."""

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

    def _is_compatible(self, operator) -> bool:
        """
        Checks whether the `operator` is compatible (same shape and

        Args:
            operator (FermionicOperator/FermionicOp): a fermionic operator

        Returns:
            True iff `operator` is compatible with `self`.
        """
        return (
            isinstance(operator, FermionicOp)
            and self.register_length == operator.register_length
        )

    def reduce(self) -> "FermionicOp":
        """
        Reduce

        Returns:
            The reduced `FermionicOp`
        """
        # TODO: atol, rtol
        label_list, indexes = np.unique(self._labels, return_inverse=True, axis=0)
        coeff_list = np.zeros(len(self._coeffs))
        for i, val in zip(indexes, self._coeffs):
            coeff_list[i] += val
        non_zero = [i for i, v in enumerate(coeff_list) if not v == 0]
        label_list = label_list[non_zero]
        coeff_list = coeff_list[non_zero]
        if not non_zero:
            return FermionicOp("I" * self.register_length, coeff=0)
        return FermionicOp(list(zip(label_list, coeff_list)))

    def __len__(self):
        return len(self._labels)

    def to_opflow(self, method: str = "JW") -> PauliSumOp:
        # TODO: other mappings
        # if method == "JW":

        # pylint: disable=cyclic-import
        from qiskit_nature.mappings.jordan_wigner_mapping import JordanWignerMapping
        return JordanWignerMapping().map(self)

    @staticmethod
    def _validate_label(label):
        return all(char in ["I", "+", "-", "N", "E"] for char in label)
