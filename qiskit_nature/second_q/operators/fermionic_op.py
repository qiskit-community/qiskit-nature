# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Fermionic-particle Operator."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping

import numpy as np
from scipy.sparse import csc_matrix

from .sparse_label_op import SparseLabelOp


class FermionicOp(SparseLabelOp):
    r"""N-mode Fermionic operator.

    A `FermionicOp` represents a weighted sum of fermionic creation/annihilation operator terms.
    These terms are encoded as strings consisting of a space-separated list of words. Each word must
    look like :code:`[+-]_<index>`, where the :code:`<index>` is a non-negative integer representing
    the index of the fermionic mode where the `+` (creation) or `-` (annihilation) operation is to
    be performed. The value of :code:`index` is bounded by the `register_length` of the operator,
    which indicates the number of fermionic modes on which the operator acts (Note: since Python
    indices are 0-based, the maximum value an index can take is given by :code:`register_length-1`).

    **Initialization**

    A `FermionicOp` is initialized with a dictionary, mapping terms to their respective
    coefficients:

    .. jupyter-execute::

        from qiskit_nature.second_q.operators import FermionicOp

        op = FermionicOp(
            {
                "+_0 -_0": 1.0,
                "+_1 -_1": -1.0,
            },
            register_length=2,
        )

    By default, this way of initializing will create a full copy of the dictionary of coefficients.
    If you have very restricted memory resources available, or would like to avoid the additional
    copy, you can disable it like so:

    .. jupyter-execute::

        some_big_data = {
            "+_0 -_0": 1.0,
            "+_1 -_1": -1.0,
            # ...
        }

        op = FermionicOp(
            some_big_data,
            register_length=2,
            copy=False,
        )


    .. note::

        It is the users responsibility, that in the above scenario, :code:`some_big_data` is not
        changed after initialization of the `FermionicOp`, since the operator contents are not
        guaranteed to remain unaffected by such changes.

    **Algebra**

    This class supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, operator multiplication, and adjoint.
    For example,

    Addition

    .. jupyter-execute::

      0.5 * FermionicOp({"+_1": 1}, register_length=2) + FermionicOp({"+_0": 1}, register_length=2)

    Sum

    .. jupyter-execute::

      0.25 * sum(FermionicOp({label: 1}, register_length=3) for label in ["+_0", "-_1", "N_2"])

    Operator multiplication

    .. jupyter-execute::

      op1 = FermionicOp({"+_0 -_1": 1}, register_length=2)
      op2 = FermionicOp({"-_0 +_0 +_1": 1}, register_length=2)
      print(op1 @ op2)

    Adjoint

    .. jupyter-execute::

      FermionicOp({"+_0 -_1": 1j}, register_length=2).adjoint()

    In principle, you can also add :class:`FermionicOp` and integers, but the only valid case is the
    addition of `0 + FermionicOp`. This makes the `sum` operation from the example above possible
    and it is useful in the following scenario:

    .. code-block:: python

        fermion = 0
        for i in some_iterable:
            # some processing
            fermion += FermionicOp(somedata)

    **Iteration**

    Instances of `FermionicOp` are iterable. Iterating a FermionicOp yields (term, coefficient)
    pairs describing the terms contained in the operator.
    """

    def __repr__(self) -> str:
        data_str = f"{dict(self.items())}"

        return "FermionicOp(" f"{data_str}, " f"register_length={self.register_length}, " ")"

    def __str__(self) -> str:
        pre = (
            "Fermionic Operator\n"
            f"register length={self.register_length}, number terms={len(self)}\n"
        )
        ret = "  " + "\n+ ".join(
            [f"{coeff} * ( {label} )" if label else f"{coeff}" for label, coeff in self.items()]
        )
        return pre + ret

    def compose(self, other: FermionicOp, qargs=None, front: bool = False) -> FermionicOp:
        if not isinstance(other, FermionicOp):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'FermionicOp' and '{type(other).__name__}'"
            )

        new_data = {
            f"{label1 if front else label2} {label2 if front else label1}".strip(): cf1 * cf2
            for label2, cf2 in other.items()
            for label1, cf1 in self.items()
        }
        register_length = max(self.register_length, other.register_length)
        return FermionicOp(new_data, register_length, copy=False)

    def tensor(self, other: FermionicOp) -> FermionicOp:
        return self._tensor(self, other)

    def expand(self, other: FermionicOp) -> FermionicOp:
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a: FermionicOp, b: FermionicOp) -> FermionicOp:
        pass

    def to_matrix(self, sparse: bool | None = True) -> csc_matrix | np.ndarray:
        """Convert to a matrix representation over the full fermionic Fock space in the occupation
        number basis.

        The basis states are ordered in increasing bitstring order as 0000, 0001, ..., 1111.

        Args:
            sparse: If true, the matrix is returned as a sparse matrix, otherwise it is returned as
                a dense numpy array.

        Returns:
            The matrix of the operator in the Fock basis
        """

        csc_data, csc_col, csc_row = [], [], []

        dimension = 1 << self.register_length

        # loop over all columns of the matrix
        for col_idx in range(dimension):
            initial_occupations = [occ == "1" for occ in f"{col_idx:0{self.register_length}b}"]
            # loop over the terms in the operator data
            for opstring, prefactor in self.simplify().items():
                # check if op string is the identity
                if not opstring:
                    csc_data.append(prefactor)
                    csc_row.append(col_idx)
                    csc_col.append(col_idx)
                else:
                    occupations = initial_occupations.copy()
                    sign = 1
                    mapped_to_zero = False

                    terms = [tuple(lbl.split("_")) for lbl in opstring.split(" ")]
                    # apply terms sequentially to the current basis state
                    for char, index in reversed(terms):
                        index = int(index)
                        occ = occupations[index]
                        if (char == "+") == occ:
                            # Applying the creation operator on an occupied state maps to zero. So
                            # does applying the annihilation operator on an unoccupied state.
                            mapped_to_zero = True
                            break
                        sign *= (-1) ** sum(occupations[:index])
                        occupations[index] = not occ

                    # add data point to matrix in the correct row
                    if not mapped_to_zero:
                        row_idx = sum(int(occ) << idx for idx, occ in enumerate(occupations[::-1]))
                        csc_data.append(sign * prefactor)
                        csc_row.append(row_idx)
                        csc_col.append(col_idx)

        sparse_mat = csc_matrix(
            (csc_data, (csc_row, csc_col)),
            shape=(dimension, dimension),
            dtype=complex,
        )

        if sparse:
            return sparse_mat
        else:
            return sparse_mat.toarray()

    def transpose(self) -> FermionicOp:
        data = {}

        trans = "".maketrans("+-", "-+")

        for label, coeff in self.items():
            data[" ".join(lbl.translate(trans) for lbl in reversed(label.split(" ")))] = coeff

        return FermionicOp(data, register_length=self.register_length, copy=False)

    def normal_ordered(self) -> FermionicOp:
        """Convert to the equivalent operator with normal order.

        Returns a new operator (the original operator is not modified).

        .. note::

            This method implements the transformation of an operator to the normal ordered operator.
            The transformation is calculated by considering all commutation relations between the
            operators.
            For example, for the case :math:`\\colon c_0 c_0^\\dagger\\colon` where :math:`c_0`
            is an annihilation operator, this method returns :math:`1 - c_0^\\dagger c_0` due to
            commutation relations.
            See the reference: https://en.wikipedia.org/wiki/Normal_order#Multiple_fermions.

        Returns:
            The normal ordered operator.
        """
        ordered_op = FermionicOp.zero(self.register_length)

        for label, coeff in self.items():
            ordered_op += self._normal_ordered(label, coeff)

        return ordered_op

    def _normal_ordered(self, label: str, coeff: complex) -> FermionicOp:
        if not label:
            return FermionicOp({"": coeff}, self.register_length)

        ordered_op = FermionicOp.zero(self.register_length)

        # 1. split label into list of pairs of the form ("char", index)
        terms = [tuple(lbl.split("_")) for lbl in label.split(" ")]

        # 2. perform insertion sorting
        for i in range(1, len(terms)):
            for j in range(i, 0, -1):
                right = terms[j]
                left = terms[j - 1]

                if right[0] == "+" and left[0] == "-":
                    # swap terms where an annihilation operator is left of a creation operator
                    terms[j - 1] = right
                    terms[j] = left
                    coeff *= -1.0

                    if right[1] == left[1]:
                        # if their indices are identical, we incur an additional term because of:
                        # a_i a_i^\dagger = 1 - a_i^\dagger a_i
                        new_label = " ".join(
                            f"{term[0]}_{term[1]}" for term in terms[: (j - 1)] + terms[(j + 1) :]
                        )
                        # we can do so by recursion on this method
                        ordered_op += self._normal_ordered(new_label, -1.0 * coeff)

                elif right[0] == left[0]:
                    # when we have identical neighboring operators, differentiate two cases:

                    # on identical index, this is an invalid Fermionic operation which evaluates to
                    # zero: e.g. +_0 +_0 = 0
                    if right[1] == left[1]:
                        # thus, we bail on this recursion call
                        return ordered_op

                    # otherwise, if the left index is higher than the right one, swap the terms
                    elif left[1] > right[1]:
                        terms[j - 1] = right
                        terms[j] = left
                        coeff *= -1.0

        new_label = " ".join(f"{term[0]}_{term[1]}" for term in terms)
        ordered_op += FermionicOp({new_label: coeff}, self.register_length, copy=False)
        return ordered_op

    def is_hermitian(self, *, atol: float | None = None) -> bool:
        """Checks whether the operator is hermitian.

        Args:
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``.

        Returns:
            True if the operator is hermitian up to numerical tolerance, False otherwise.
        """
        atol = self.atol if atol is None else atol
        diff = (self - self.adjoint()).normal_ordered().simplify(atol=atol)
        return all(np.isclose(coeff, 0.0, atol=atol) for coeff in diff.values())

    def simplify(self, *, atol: float | None = None) -> FermionicOp:
        """Simplify the operator.

        Merges terms with same labels and eliminates terms with coefficients close to 0.
        Returns a new operator (the original operator is not modified).

        Args:
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``.

        Returns:
            The simplified operator.
        """
        atol = self.atol if atol is None else atol

        data = defaultdict(complex)  # type: dict[str, complex]
        # TODO: use parallel_map to make this more efficient (?)
        for label, coeff in self.items():
            label, coeff = self._simplify_label(label, coeff)
            data[label] += coeff
        simplified_data = {
            label: coeff for label, coeff in data.items() if not np.isclose(coeff, 0.0, atol=atol)
        }
        return FermionicOp(simplified_data, self.register_length, copy=False)

    def _simplify_label(self, label: str, coeff: complex) -> tuple[str, complex]:
        bits = _BitsContainer()

        for lbl in label.split():
            char, index = lbl.split("_")
            idx = int(index)
            char_b = char == "+"

            if idx not in bits:
                bits[idx] = int(f"{char_b:b}{not char_b:b}{char_b:b}{char_b:b}", base=2)
                # we store all relevant information for each register index in 4 bits:
                #   1. True if a `+` has been applied on this index
                #   2. True if a `-` has been applied on this index
                #   3. True if a `+` was applied first, False if a `-` was applied first
                #   4. True if the last added operation on this index was `+`, False if `-`

            elif bits.get_last(idx) == char_b:
                # we bail, if we apply the same operator as the last one
                return "", 0

            elif bits.get_plus(idx) and bits.get_minus(idx):
                # if both, `+` and `-`, have already been applied, we cancel the opposite to the
                # current one (i.e. `+` will cancel `-` and vice versa)
                bits.set_plus_or_minus(idx, not char_b, False)
                # we also update the last bit to the current char
                bits.set_last(idx, char_b)

            else:
                # else, we simply set the bit of the currently applied char
                bits.set_plus_or_minus(idx, char_b, True)
                # we also update the last bit to the current char
                bits.set_last(idx, char_b)

            if idx != self.register_length:
                num_exchange = 0
                for i in range(idx + 1, self.register_length):
                    if i in bits:
                        num_exchange += (bits.get_plus(i) + bits.get_minus(i)) % 2
                coeff *= -1 if num_exchange % 2 else 1

        new_label = []
        for idx in sorted(bits):
            plus = f"+_{idx}" if bits.get_plus(idx) else None
            minus = f"-_{idx}" if bits.get_minus(idx) else None
            new_label.extend([plus, minus] if bits.get_order(idx) else [minus, plus])

        return " ".join(lbl for lbl in new_label if lbl is not None), coeff


class _BitsContainer(MutableMapping):
    """A bit-storage container.

    This is a utility object used during the simplification process of a `FermionicOp`.
    It manages access to an internal data container, which maps from integers to bytes.
    Each integer key corresponds to a fermionic mode of an operator term.
    Each value consists of four bits which encoding for the corresponding index:

        1. if a `+` has been applied
        2. if a `-` has been applied
        3. whether a `+` or `-` was applied first
        4. whether the last applied operator was a `+` or `-`
    """

    def __init__(self):
        self.data: dict[int, int] = {}

    def get_plus(self, index: int) -> int:
        """Returns the value of the `+`-register.

        Args:
            index: the internal data key (corresponding to the fermionic mode).

        Returns:
            1 if `+` has been applied, 0 otherwise.
        """
        return self.get_bit(index, 3)

    def get_minus(self, index: int) -> int:
        """Returns the value of the `-`-register.

        Args:
            index: the internal data key (corresponding to the fermionic mode).

        Returns:
            1 if `-` has been applied, 0 otherwise.
        """
        return self.get_bit(index, 2)

    def set_plus_or_minus(self, index: int, plus_or_minus: bool, value: bool) -> None:
        """Sets the `+`- or `-`-register of the provided index to the provided value.

        Args:
            index: the internal data key (corresponding to the fermionic mode).
            plus_or_minus: True if the `+`-register is to be set, False for the `-`-register
            value: True if the register is to be set to 1, False for 0.
        """
        if value:
            # plus is stored at index 0, but plus_or_minus is True if it is Plus
            self.set_bit(index, 3 - int(not plus_or_minus))
        else:
            self.clear_bit(index, 3 - int(not plus_or_minus))

    def get_order(self, index: int) -> int:
        """Returns the value of the order-register.

        Note: the order-register is read-only and can only be set during initialization.

        Args:
            index: the internal data key (corresponding to the fermionic mode).

        Returns:
            1 if `+` was applied first, 0 if `-` was applied first.
        """
        return self.get_bit(index, 1)

    def get_last(self, index: int) -> int:
        """Returns the value of the last-register.

        Args:
            index: the internal data key (corresponding to the fermionic mode).

        Returns:
            1 if `+` was applied last, 0 otherwise.
        """
        return self.get_bit(index, 0)

    def set_last(self, index: int, value: bool) -> None:
        """Sets the value of the last-register.

        Args:
            index: the internal data key (corresponding to the fermionic mode).
            value: True if the register is to be set to 1, False for 0.
        """
        if value:
            self.set_bit(index, 0)
        else:
            self.clear_bit(index, 0)

    def get_bit(self, index: int, offset: int) -> int:
        """Returns the value of a requested register.

        Args:
            index: the internal data key (corresponding to the fermionic mode).
            offset: the bit-wise offset for the bit-shift operation to obtain the desired register.

        Returns:
            1 if the register was set, 0 otherwise.
        """
        return (self.data[index] >> offset) & 1

    def set_bit(self, index: int, offset: int) -> None:
        """Sets the provided register to 1.

        Args:
            index: the internal data key (corresponding to the fermionic mode).
            offset: the bit-wise offset for the bit-shift operation to set the desired register.
        """
        self.data[index] = self.data[index] | (1 << offset)

    def clear_bit(self, index: int, offset: int) -> None:
        """Clears the provided register (to 0).

        Args:
            index: the internal data key (corresponding to the fermionic mode).
            offset: the bit-wise offset for the bit-shift operation to set the desired register.
        """
        self.data[index] = self.data[index] & ~(1 << offset)

    def __getitem__(self, __k):
        return self.data.__getitem__(__k)

    def __setitem__(self, __k, __v):
        return self.data.__setitem__(__k, __v)

    def __delitem__(self, __v):
        return self.data.__delitem__(__v)

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return self.data.__len__()
