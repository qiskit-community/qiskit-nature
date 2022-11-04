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

import re
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import cast, Iterator

import numpy as np
from scipy.sparse import csc_matrix

from qiskit_nature.exceptions import QiskitNatureError
import qiskit_nature.optionals as _optionals

from ._bits_container import _BitsContainer
from .polynomial_tensor import PolynomialTensor
from .sparse_label_op import _TCoeff, SparseLabelOp, _to_number


class FermionicOp(SparseLabelOp):
    r"""N-mode Fermionic operator.

    A ``FermionicOp`` represents a weighted sum of fermionic creation/annihilation operator terms.
    These terms are encoded as sparse labels, which are strings consisting of a space-separated list of
    expressions. Each expression must look like :code:`[+-]_<index>`, where the :code:`<index>` is a
    non-negative integer representing the index of the fermionic mode where the ``+`` (creation) or
    ``-`` (annihilation) operation is to be performed. The value of :code:`index` is bound by the
    number of spin orbitals (``num_spin_orbitals``) of the operator (Note: since Python indices are
    0-based, the maximum value an index can take is given by :code:`num_spin_orbitals-1`).

    **Initialization**

    A ``FermionicOp`` is initialized with a dictionary, mapping terms to their respective
    coefficients:

    .. jupyter-execute::

        from qiskit_nature.second_q.operators import FermionicOp

        op = FermionicOp(
            {
                "+_0 -_0": 1.0,
                "+_1 -_1": -1.0,
            },
            num_spin_orbitals=2,
        )

    By default, this way of initializing will create a full copy of the dictionary of coefficients.
    If you have very restricted memory resources available, or would like to avoid the additional
    copy, the dictionary will be stored by reference if you disable ``copy`` like so:

    .. jupyter-execute::

        some_big_data = {
            "+_0 -_0": 1.0,
            "+_1 -_1": -1.0,
            # ...
        }

        op = FermionicOp(
            some_big_data,
            num_spin_orbitals=2,
            copy=False,
        )


    .. note::

        It is the users' responsibility, that in the above scenario, :code:`some_big_data` is not
        changed after initialization of the `FermionicOp`, since the operator contents are not
        guaranteed to remain unaffected by such changes.

    **Algebra**

    This class supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, operator multiplication, and adjoint.
    For example,

    Addition

    .. jupyter-execute::

      FermionicOp({"+_1": 1}, num_spin_orbitals=2) + FermionicOp({"+_0": 1}, num_spin_orbitals=2)

    Sum

    .. jupyter-execute::

      sum(FermionicOp({label: 1}, num_spin_orbitals=3) for label in ["+_0", "-_1", "+_2 -_2"])

    Scalar multiplication

    .. jupyter-execute::

      0.5 * FermionicOp({"+_1": 1}, num_spin_orbitals=2)

    Operator multiplication

    .. jupyter-execute::

      op1 = FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2)
      op2 = FermionicOp({"-_0 +_0 +_1": 1}, num_spin_orbitals=2)
      print(op1 @ op2)

    Tensor multiplication

    .. jupyter-execute::

      op = FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2)
      print(op ^ op)

    Adjoint

    .. jupyter-execute::

      FermionicOp({"+_0 -_1": 1j}, num_spin_orbitals=2).adjoint()

    In principle, you can also add `FermionicOp` and integers, but the only valid case is the
    addition of `0 + FermionicOp`. This makes the `sum` operation from the example above possible
    and it is useful in the following scenario:

    .. code-block:: python

        fermion = 0
        for i in some_iterable:
            # some processing
            fermion += FermionicOp(somedata)

    **Iteration**

    Instances of ``FermionicOp`` are iterable. Iterating a ``FermionicOp`` yields (term, coefficient)
    pairs describing the terms contained in the operator.

    Attributes:
        num_spin_orbitals: the number of spin orbitals on which this operator acts. This is
            considered a lower bound, which means that mathematical operations acting on two or more
            operators will result in a new operator with the maximum number of spin orbitals of any
            of the involved operators.

    .. note::

        A FermionicOp can contain :class:`qiskit.circuit.ParameterExpression` objects as coefficients.
        However, a FermionicOp containing parameters does not support the following methods:

        - ``is_hermitian``
        - ``to_matrix``
    """

    _OPERATION_REGEX = re.compile(r"([\+\-]_\d+\s)*[\+\-]_\d+")

    def __init__(
        self,
        data: Mapping[str, _TCoeff],
        num_spin_orbitals: int | None = None,
        *,
        copy: bool = True,
        validate: bool = True,
    ) -> None:
        """
        Args:
            data: the operator data, mapping string-based keys to numerical values.
            num_spin_orbitals: the number of spin orbitals on which this operator acts.
            copy: when set to False the ``data`` will not be copied and the dictionary will be
                stored by reference rather than by value (which is the default; ``copy=True``). Note,
                that this requires you to not change the contents of the dictionary after
                constructing the operator. This also implies ``validate=False``. Use with care!
            validate: when set to False the ``data`` keys will not be validated. Note, that the
                SparseLabelOp base class, makes no assumption about the data keys, so will not
                perform any validation by itself. Only concrete subclasses are encouraged to
                implement a key validation method. Disable this setting with care!

        Raises:
            QiskitNatureError: when an invalid key is encountered during validation.
        """
        self.num_spin_orbitals = num_spin_orbitals
        super().__init__(data, copy=copy, validate=validate)

    @property
    def register_length(self) -> int | None:
        return self.num_spin_orbitals

    def _new_instance(
        self, data: Mapping[str, _TCoeff], *, other: FermionicOp | None = None
    ) -> FermionicOp:
        num_so = self.num_spin_orbitals
        if other is not None:
            other_num_so = other.num_spin_orbitals
            if num_so is None:
                num_so = other_num_so
            elif other_num_so is not None:
                num_so = max(num_so, other_num_so)

        return self.__class__(data, copy=False, num_spin_orbitals=num_so)

    def _validate_keys(self, keys: Collection[str]) -> None:
        super()._validate_keys(keys)

        num_so = self.num_spin_orbitals

        max_index = -1

        for key in keys:
            # 0. explicitly allow the empty key
            if key == "":
                continue

            # 1. validate overall key structure
            if not re.fullmatch(FermionicOp._OPERATION_REGEX, key):
                raise QiskitNatureError(f"{key} is not a valid FermionicOp label.")

            # 2. validate all indices against register length
            for term in key.split():
                index = int(term[2:])
                if num_so is None:
                    if index > max_index:
                        max_index = index
                elif index >= num_so:
                    raise QiskitNatureError(
                        f"The index, {index}, from the label, {key}, exceeds the number of spin "
                        f"orbitals, {num_so}."
                    )

        self.num_spin_orbitals = max_index + 1 if num_so is None else num_so

    @classmethod
    def _validate_polynomial_tensor_key(cls, keys: Collection[str]) -> None:
        allowed_chars = {"+", "-"}

        for key in keys:
            if set(key) - allowed_chars:
                raise QiskitNatureError(
                    f"The key {key} is invalid. PolynomialTensor keys may only consists of `+` and "
                    "`-` characters, for them to be expandable into a FermionicOp."
                )

    @classmethod
    def from_polynomial_tensor(cls, tensor: PolynomialTensor) -> FermionicOp:
        cls._validate_polynomial_tensor_key(tensor.keys())

        data: dict[str, _TCoeff] = {}

        for key in tensor:
            if key == "":
                # TODO: deal with complexity
                data[""] = cast(float, tensor[key])
                continue

            label_template = " ".join(f"{op}_{{}}" for op in key)

            # PERF: the following matrix unpacking is a performance bottleneck!
            # We could consider using Rust in the future to improve upon this.

            mat = tensor[key]
            if isinstance(mat, np.ndarray):
                for index in np.ndindex(*mat.shape):
                    data[label_template.format(*index)] = mat[index]
            else:
                _optionals.HAS_SPARSE.require_now("SparseArray")
                import sparse as sp  # pylint: disable=import-error

                if isinstance(mat, sp.SparseArray):
                    coo = sp.as_coo(mat)
                    for value, *index in zip(coo.data, *coo.coords):
                        data[label_template.format(*index)] = value

        return cls(data, copy=False, num_spin_orbitals=tensor.register_length).chop()

    def __repr__(self) -> str:
        data_str = f"{dict(self.items())}"

        return "FermionicOp(" f"{data_str}, " f"num_spin_orbitals={self.num_spin_orbitals}, " ")"

    def __str__(self) -> str:
        pre = (
            "Fermionic Operator\n"
            f"number spin orbitals={self.num_spin_orbitals}, number terms={len(self)}\n"
        )
        ret = "  " + "\n+ ".join(
            [f"{coeff} * ( {label} )" if label else f"{coeff}" for label, coeff in self.items()]
        )
        return pre + ret

    def terms(self) -> Iterator[tuple[list[tuple[str, int]], _TCoeff]]:
        """Provides an iterator analogous to :meth:`items` but with the labels already split into
        pairs of operation characters and indices.

        Yields:
            A tuple with two items; the first one being a list of pairs of the form (char, int)
            where char is either `+` or `-` and the integer corresponds to the fermionic mode index
            on which the operator gets applied; the second item of the returned tuple is the
            coefficient of this term.
        """
        for label in iter(self):
            if not label:
                yield ([], self[label])
                continue
            # we hard-code the result of lbl.split("_") as follows:
            #   lbl[0] is either + or -
            #   lbl[2:] corresponds to the index
            terms = [(lbl[0], int(lbl[2:])) for lbl in label.split()]
            yield (terms, self[label])

    def compose(self, other: FermionicOp, qargs=None, front: bool = False) -> FermionicOp:
        if not isinstance(other, FermionicOp):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'FermionicOp' and '{type(other).__name__}'"
            )

        if front:
            return self._tensor(self, other, offset=False)
        else:
            return self._tensor(other, self, offset=False)

    def tensor(self, other: FermionicOp) -> FermionicOp:
        return self._tensor(self, other)

    def expand(self, other: FermionicOp) -> FermionicOp:
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a: FermionicOp, b: FermionicOp, *, offset: bool = True) -> FermionicOp:
        shift = a.num_spin_orbitals if offset else 0

        new_data: dict[str, _TCoeff] = {}
        for label1, cf1 in a.items():
            for terms2, cf2 in b.terms():
                new_label = f"{label1} {' '.join(f'{c}_{i+shift}' for c, i in terms2)}".strip()
                if new_label in new_data:
                    new_data[new_label] += cf1 * cf2
                else:
                    new_data[new_label] = cf1 * cf2

        new_op = a._new_instance(new_data, other=b)
        if offset:
            new_op.num_spin_orbitals = a.num_spin_orbitals + b.num_spin_orbitals
        return new_op

    # TODO: do we want to change the returned type to be non-scipy sparse matrix?
    def to_matrix(self, sparse: bool | None = True) -> csc_matrix | np.ndarray:
        """Convert to a matrix representation over the full fermionic Fock space in the occupation
        number basis.

        The basis states are ordered in increasing bitstring order as 0000, 0001, ..., 1111.

        Args:
            sparse: If true, the matrix is returned as a sparse matrix, otherwise it is returned as
                a dense numpy array.

        Returns:
            The matrix of the operator in the Fock basis

        Raises:
            ValueError: Operator contains parameters.
        """
        if self.is_parameterized():
            raise ValueError("to_matrix is not supported for operators containing parameters.")

        csc_data, csc_col, csc_row = [], [], []

        dimension = 1 << self.num_spin_orbitals

        # loop over all columns of the matrix
        for col_idx in range(dimension):
            initial_occupations = [occ == "1" for occ in f"{col_idx:0{self.num_spin_orbitals}b}"]
            # loop over the terms in the operator data
            for terms, prefactor in self.simplify().terms():
                # check if op string is the identity
                if not terms:
                    csc_data.append(prefactor)
                    csc_row.append(col_idx)
                    csc_col.append(col_idx)
                else:
                    occupations = initial_occupations.copy()
                    sign = 1
                    mapped_to_zero = False

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
            data[" ".join(lbl.translate(trans) for lbl in reversed(label.split()))] = coeff

        return self._new_instance(data)

    def normal_order(self) -> FermionicOp:
        """Convert to the equivalent operator in normal order.

        The normal order for fermions is defined
        [here](https://en.wikipedia.org/wiki/Normal_order#Fermions).

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
        ordered_op = FermionicOp.zero()

        for terms, coeff in self.terms():
            ordered_op += self._normal_order(terms, coeff)

        # after successful normal ordering, we remove all zero coefficients
        return self._new_instance(
            {
                label: coeff
                for label, coeff in ordered_op.items()
                if not np.isclose(_to_number(coeff), 0.0, atol=self.atol)
            }
        )

    def _normal_order(self, terms: list[tuple[str, int]], coeff: _TCoeff) -> FermionicOp:
        if not terms:
            return self._new_instance({"": coeff})

        ordered_op = FermionicOp.zero()

        # perform insertion sorting
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
                        new_terms = terms[: (j - 1)] + terms[(j + 1) :]
                        # we can do so by recursion on this method
                        ordered_op += self._normal_order(new_terms, -1.0 * coeff)

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
        ordered_op += self._new_instance({new_label: coeff})
        return ordered_op

    def index_order(self) -> FermionicOp:
        """Convert to the equivalent operator with the terms of each label ordered by index.

        Returns a new operator (the original operator is not modified).

        .. note::

            You can use this method to achieve the most aggressive simplification of an operator
            without changing the operation order per index. :meth:`simplify` does *not* reorder the
            terms and, thus, cannot deduce ``-_0 +_1`` and ``+_1 -_0 +_0 -_0`` to be
            identical labels. Calling this method will reorder the latter label to
            ``-_0 +_0 -_0 +_1``, after which :meth:`simplify` will be able to correctly collapse
            these two labels into one.

        Returns:
            The index ordered operator.
        """
        data = defaultdict(complex)  # type: dict[str, _TCoeff]
        for terms, coeff in self.terms():
            label, coeff = self._index_order(terms, coeff)
            data[label] += coeff

        # after successful index ordering, we remove all zero coefficients
        return self._new_instance(
            {
                label: coeff
                for label, coeff in data.items()
                if not np.isclose(_to_number(coeff), 0.0, atol=self.atol)
            }
        )

    def _index_order(self, terms: list[tuple[str, int]], coeff: _TCoeff) -> tuple[str, _TCoeff]:
        if not terms:
            return "", coeff

        # perform insertion sorting
        for i in range(1, len(terms)):
            for j in range(i, 0, -1):
                right = terms[j]
                left = terms[j - 1]

                if left[1] > right[1]:
                    terms[j - 1] = right
                    terms[j] = left
                    coeff *= -1.0

        new_label = " ".join(f"{term[0]}_{term[1]}" for term in terms)
        return new_label, coeff

    def is_hermitian(self, atol: float | None = None) -> bool:
        """Checks whether the operator is hermitian.

        Args:
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``.

        Returns:
            True if the operator is hermitian up to numerical tolerance, False otherwise.

        Raises:
            ValueError: Operator contains parameters.
        """
        if self.is_parameterized():
            raise ValueError("is_hermitian is not supported for operators containing parameters.")
        atol = self.atol if atol is None else atol
        diff = (self - self.adjoint()).normal_order().simplify(atol=atol)
        return all(np.isclose(coeff, 0.0, atol=atol) for coeff in diff.values())

    def simplify(self, atol: float | None = None) -> FermionicOp:
        atol = self.atol if atol is None else atol

        data = defaultdict(complex)  # type: dict[str, _TCoeff]
        # TODO: use parallel_map to make this more efficient (?)
        for label, coeff in self.items():
            label, coeff = self._simplify_label(label, coeff)
            data[label] += coeff
        simplified_data = {
            label: coeff
            for label, coeff in data.items()
            if not np.isclose(_to_number(coeff), 0.0, atol=atol)
        }
        return self._new_instance(simplified_data)

    def _simplify_label(self, label: str, coeff: _TCoeff) -> tuple[str, _TCoeff]:
        bits = _BitsContainer[int]()

        # Since Python 3.7, dictionaries are guaranteed to be insert-order preserving. We use this
        # to our advantage, to implement an ordered set, which allows us to preserve the label order
        # and only remove canceling terms.
        new_label: dict[str, None] = {}

        for lbl in label.split():
            char, index = lbl.split("_")
            idx = int(index)
            char_b = char == "+"

            if idx not in bits:
                # we store all relevant information for each register index in 4 bits:
                #   1. True if a `+` has been applied on this index
                #   2. True if a `-` has been applied on this index
                #   3. True if a `+` was applied first, False if a `-` was applied first
                #   4. True if the last added operation on this index was `+`, False if `-`
                bits[idx] = int(f"{char_b:b}{not char_b:b}{char_b:b}{char_b:b}", base=2)
                # and we insert the encountered label into our ordered set
                new_label[lbl] = None

            elif bits.get_last(idx) == char_b:
                # we bail, if we apply the same operator as the last one
                return "", 0

            elif bits.get_plus(idx) and bits.get_minus(idx):
                # If both, `+` and `-`, have already been applied, we cancel the opposite to the
                # current one (i.e. `+` will cancel `-` and vice versa).
                # 1. we construct the reversed label which is the key we need to pop
                pop_lbl = f"{'-' if char_b else '+'}_{idx}"
                # 2. we find its index in the insertion order of the new label
                pop_idx = list(new_label).index(pop_lbl)
                # 3. we use this index plus the current length of the new label to determine the
                #    number of exchange operations necessary to move the current term next to the
                #    one being popped
                num_exchange = len(new_label) - pop_idx - 1
                # 4. we perform the information update by:
                #  a) updating the coefficient sign
                coeff *= -1 if num_exchange % 2 else 1
                #  b) popping the key we just canceled out
                new_label.pop(pop_lbl)
                #  c) updating the bits container
                bits.set_plus_or_minus(idx, not char_b, False)
                #  d) and updating the last bit to the current char
                bits.set_last(idx, char_b)

            else:
                # else, we simply set the bit of the currently applied char
                bits.set_plus_or_minus(idx, char_b, True)
                # and track it in our ordered set
                new_label[lbl] = None
                # we also update the last bit to the current char
                bits.set_last(idx, char_b)

        return " ".join(new_label), coeff
