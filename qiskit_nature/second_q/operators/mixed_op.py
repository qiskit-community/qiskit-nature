# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Mixed Operator class."""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Callable

from qiskit.quantum_info.operators.mixins import LinearMixin

from .sparse_label_op import SparseLabelOp


class MixedOp(LinearMixin):
    """Mixed operator.

    A ``MixedOp`` represents a weighted sum of products of fermionic/bosonic operators potentially
    acting on different "local" Hilbert spaces. The terms to be summed are encoded in a dictionary,
    where each operator product is identified by its key, a tuple of string specifying the names of the
    local Hilbert spaces on which it acts, and by its value, a list of tuple (corresponding to a sum of
    operators acting on the same composite Hilbert space) where each tuple encodes the coupling
    coefficient and the operators themselves (that might also have coefficients associated with them).


    **Initialization**

    A ``MixedOp`` is initialized with a dictionary, mapping terms to their respective
    coefficients:

    .. code-block:: python

        from qiskit_nature.second_q.operators import FermionicOp, SpinOp, MixedOp

        fop1 = FermionicOp({"+_0 -_0": 1}) # Acting on Hilbert space "h1"
        sop1 = SpinOp({"X_0 Y_0": 1}, num_spins=1) # Acting on Hilbert space "s1"

        mop1 = MixedOp({("h1",): [(5.0, fop1)]}) # 5.0 * fop1
        mop2 = MixedOp(
            {
                ("h1", "s1"): [(3, fop1, sop1)],
                ("s1",): [(2, sop1)],
            }
        ) # 3*(fop1 @ sop1) + 2*(sop1)


    **Algebra**

    This class supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, operator multiplication.
    For example,

    Addition

    .. code-block:: python

        fop1 = FermionicOp({"+_0 -_0": 1}) # Acting on Hilbert space "h1"
        sop1 = SpinOp({"X_0 Y_0": 1}, num_spins=1) # Acting on Hilbert space "s1"
        MixedOp({("h1",): [(5.0, fop1)]}) + MixedOp({("s1",): [(6.0, sop1)]})
        # MixedOp({("h1",): [(5.0, fop1)], ("s1",): [(6.0, sop1)]})

    Scalar multiplication

    .. code-block:: python

        fop1 = FermionicOp({"+_0 -_0": 1})
        0.5 * MixedOp({("h1",): [(5.0, fop1)]})

    Operator multiplication

    .. code-block:: python

        fop1 = FermionicOp({"+_0 -_0": 1}) # Acting on Hilbert space "h1"
        sop1 = SpinOp({"X_0 Y_0": 1}, num_spins=1) # Acting on Hilbert space "s1"
        MixedOp({("h1",): [(5.0, fop1)]}) @ MixedOp({("s1",): [(6.0, sop1)]})
        # MixedOp({("h1", "s1"): [(30.0, fop1, sop1)]})

    """

    def __init__(self, data: dict[tuple[str], list[tuple[float, SparseLabelOp]]]):
        self.data = deepcopy(data)

    def __repr__(self, indentation_level=0) -> str:
        out_str = "Mixed Op\n"
        out_str += f"Nb terms = {len(self.data)}\n"
        for active_indices, oplist in self.data.items():
            for op_tuple in oplist:
                coef, active_operators = op_tuple[0], op_tuple[1:]
                out_str += f"- Coefficient: {coef:.02f}\n"

                for index, op in zip(active_indices, active_operators):
                    out_str += "" * indentation_level + f"{index}: {repr(op)}\n"

                out_str += "\n"

        return out_str

    @staticmethod
    def _tuple_prod(tup1: tuple[int, ...], tup2: tuple) -> tuple[float, ...]:
        """Implements the composition of operator tuples representing tensor products of operators."""
        new_coeff = tup1[0] * tup2[0]
        new_op_tuple = tup1[1:] + tup2[1:]
        return (new_coeff,) + new_op_tuple

    @staticmethod
    def _tuple_multiply(tup: tuple[int, ...], coef: float) -> tuple[float, ...]:
        """Implements the dilation by a coefficient of an operator tuple representing a tensor product
        of operators."""
        new_coeff = tup[0] * coef
        return (new_coeff,) + tup[1:]

    @classmethod
    def _distribute_on_tuples(
        cls, method: Callable, op_left: MixedOp, op_right: MixedOp = None, **kwargs
    ) -> MixedOp:
        """Implements the distributions of a method to the tuples of operators representing the product
        of operators."""
        new_op_data: dict = {}
        if op_right is None:
            # Distribute method over all tuples.
            for key, op_tuple_list in op_left.data.items():
                new_op_data[key] = [method(op_tuple, **kwargs) for op_tuple in op_tuple_list]
        else:
            # Distribute method over all combinations of tuples from the first and second operators.
            for (key1, op_tuple_list1), (key2, op_tuple_list2) in itertools.product(
                op_left.data.items(), op_right.data.items()
            ):
                new_op_data[key1 + key2] = [
                    method(op_tuple1, op_tuple2, **kwargs)
                    for (op_tuple1, op_tuple2) in itertools.product(op_tuple_list1, op_tuple_list2)
                ]

        return MixedOp(new_op_data)

    def _multiply(self, other: float) -> MixedOp:
        """Return Operator multiplication of self and other.

        Args:
            other: the second ``MixedOp`` to multiply to the first.
            qargs: UNUSED.

        Returns:
            The new multiplied ``MixedOp``.
        """
        return MixedOp._distribute_on_tuples(MixedOp._tuple_multiply, op_left=self, coef=other)

    def _add(self, other: MixedOp, qargs: None = None) -> MixedOp:
        """Return Operator addition of self and other.

        Args:
            other: the second ``MixedOp`` to add to the first.
            qargs: UNUSED.

        Returns:
            The new summed ``MixedOp``.
        """

        sum_op = MixedOp(self.data)  # deepcopy
        for key in other.data.keys():
            # If the key for the composite Hilbert space already exists in the dictionary, then the
            # addition is performed by appending the new operator to the corresponding list.
            # Otherwise, the addition is performed by adding a new pair key, value to the dictionary.
            if key in sum_op.data.keys():
                sum_op.data[key] += other.data[key]
            else:
                sum_op.data[key] = other.data[key]
        return sum_op

    @classmethod
    def compose(cls, op_left: MixedOp, op_right: MixedOp) -> MixedOp:
        """Returns Operator composition of self and other.

        Args:
            op_left: left MixedOp to tensor.
            op_right: right MixedOp to tensor.

        Returns:
            The tensor product of left with right.
        """

        # Lazy composition without applying the products.
        return MixedOp._distribute_on_tuples(
            MixedOp._tuple_prod, op_left=op_left, op_right=op_right
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.data == other.data
