from __future__ import annotations

from .sparse_label_op import SparseLabelOp
from .fermionic_op import FermionicOp
from .spin_op import SpinOp
from typing import cast

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping

from qiskit.quantum_info.operators.mixins import (
    AdjointMixin,
    GroupMixin,
    LinearMixin,
    TolerancesMixin,
)

from copy import deepcopy
import numpy as np
import itertools


class MixedOp(LinearMixin):
    """Mixed operator.

    A ``MixedOp`` represents a weighted sum of products of fermionic/bosonic operators potentially
    acting on different "local" Hilbert spaces. The terms to be summed are encoded in a dictionary,
    where each operator product is identified by its key, a tuple of string specifying the names of the
    local Hilbert spaces on which it acts, and by its value, a list of tuple (corresponding to a sum of
    operators acting on the same composite Hilbert space) where each tuple encodes the coupling
    coefficient and the operators themselves (that might also have coefficients asscociated with them).


    **Initialization**

    A ``MixedOp`` is initialized with a dictionary, mapping terms to their respective
    coefficients:

    .. code-block:: python

    from qiskit_nature.second_q.operators import FermionicOp, MixedOp

    fop1 = FermionicOp({"+_0 -_0": 1}) # Acting on Hilbert space "h1"
    sop1 = SpinOp({"X_0 Y_0": 1}, num_spins=1) # Acting on Hilbert space "s1"

    mop1 = MixedOp({("h1",): [(5.0, fop1)]}) # 5.0 * fop1
    mop2 = MixedOp(
        {
            ("h1", "s1"): [(3, fop1, sop1)],
            ("s1",): [(2, sop1)],
        }
    ) # 3*(fop1 @ sop1) + 2*(sop1)


    .. note::
    Note that the ``MixedOp`` objcet can be initialized without even knowing the structure of the
    "global" Hilbert space of the problem. However, the user is expected to use an unambiguous naming
    convention for the "local" Hilbert spaces to ensure a coherent construction.
    A more precise specification of the "global' Hilbert space will be required for the mapping of the
    ``MixedOp`` to qubit operators, setting the desired ordering and size of the registers corresponding
    to each "local" Hilbert spaces.

    **Algebra**

    This class supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, operator multiplication.
    For example,

    Addition

    .. code-block:: python

        fop1 = FermionicOp({"+_0 -_0": 1}) # Acting on Hilbert space "h1"
        sop1 = SpinOp({"X_0 Y_0": 1}, num_spins=1) # Acting on Hilbert space "s1"
        MixedOp({("h1",): [(5.0, fop1)]}) + MixedOp({("s1",): [(6.0, sop1)]})

      MixedOp({("h1",): [(5.0, fop1)], ("s1",): [(6.0, sop1)]})

    Scalar multiplication

    .. code-block:: python

        fop1 = FermionicOp({"+_0 -_0": 1})
        0.5 * MixedOp({("h1",): [(5.0, fop1)]})

    Operator multiplication

    .. code-block:: python

        fop1 = FermionicOp({"+_0 -_0": 1}) # Acting on Hilbert space "h1"
        sop1 = SpinOp({"X_0 Y_0": 1}, num_spins=1) # Acting on Hilbert space "s1"
        MixedOp({("h1",): [(5.0, fop1)]}) @ MixedOp({("s1",): [(6.0, sop1)]})

      MixedOp({("h1", "s1"): [(30.0, fop1, sop1)]})
    """

    def __init__(self, data=dict[tuple, list[tuple[SparseLabelOp, float]]]):
        self.data = deepcopy(data)

    def __repr__(self) -> str:
        out_str = ""
        for key, oplist in self.data.items():
            for hspace in key:
                out_str += hspace
            out_str += " : "
            for op_tuple in oplist:
                out_str += "["
                for op in op_tuple:
                    out_str += f"{repr(op)} "
                out_str += "]"
            out_str += "\n"

        return out_str

    def keys_no_duplicate(self):
        all_keys = ()
        for key in self.data.keys():
            all_keys += key
        return tuple(set(all_keys))

    @staticmethod
    def _tuple_prod(tup1, tup2):
        """Implements the composition of operator tuples representing tensor products of operators."""
        new_coeff = tup1[0] * tup2[0]
        new_op_tuple = tup1[1:] + tup2[1:]
        return (new_coeff,) + new_op_tuple

    @staticmethod
    def _tuple_multiply(tup, coef):
        """Implements the dilation by a coefficient of an operator tuple representing a tensor product
        of operators."""
        new_coeff = tup[0] * coef
        return (new_coeff,) + tup[1:]

    @staticmethod
    def _tuple_conjugate(tup):
        """Implements the conjugaison of an operator tuple representing a tensor product of operators."""
        new_coeff = np.conjugate(tup[0])
        new_op_tuple = tuple(op.conjugate() for op in tup[1:])
        return (new_coeff,) + new_op_tuple

    @staticmethod
    def _tuple_transpose(tup):
        """Implements the trasnsposition of an operator tuple representing a tensor product of
        operators."""
        new_coeff = tup[0]
        new_op_tuple = tuple(op.transpose() for op in tup[1:])
        return (new_coeff,) + new_op_tuple

    @staticmethod
    def _tuple_adjoint(tup):
        """Implements the adjoint of an operator tuple representing a tensor product of operators."""
        new_coeff = np.conjugate(tup[0])
        new_op_tuple = tuple(op.adjoint() for op in tup[1:])
        return (new_coeff,) + new_op_tuple

    def _apply_on_tuples(self, method, *args, **kwargs) -> MixedOp:
        new_op_data = {}
        for key, op_tuple_list in self.data.items():
            new_op_data[key] = [
                method(op_tuple, *args, **kwargs) for op_tuple in op_tuple_list
            ]
        return MixedOp(new_op_data)

    def conjugate(self):
        """Returns the conjugate of the operator."""
        return self._apply_on_tuples(MixedOp._tuple_conjugate)

    def transpose(self):
        """Returns the transpose of the operator."""
        return self._apply_on_tuples(MixedOp._tuple_transpose)

    def adjoint(self):
        """Returns the adjoint of the operator."""
        return self._apply_on_tuples(MixedOp._tuple_adjoint)

    def _multiply(self, other: float):
        """Return Operator multiplication of self and other.

        Args:
            other: the second ``MixedOp`` to multiply to the first.
            qargs: UNUSED.

        Returns:
            The new multiplied ``MixedOp``.
        """
        return self._apply_on_tuples(MixedOp._tuple_multiply, other)

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

    def compose(self, other: MixedOp) -> MixedOp:
        """Returns Operator composition of self and other.

        Args:
            other: the other operator.
            qargs: UNUSED.

        Returns:
            The operator resulting from the composition.
        """

        # Lazy composition without applying the products.
        new_data = {}
        for key1, key2 in itertools.product(self.data.keys(), other.data.keys()):
            op_tuple_list1, op_tuple_list2 = self.data[key1], other.data[key2]
            new_data[key1 + key2] = [
                MixedOp._tuple_prod(op_tuple1, op_tuple2)
                for op_tuple1, op_tuple2 in itertools.product(
                    op_tuple_list1, op_tuple_list2
                )
            ]
        return MixedOp(new_data)

    def __eq__(self, other):
        if self.data.keys() != other.data.keys():
            return False
        return all(
            [self.data[key] == other.data[key] for key in self.data.keys()]
        )
