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

"""The Vibrational Operator."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Mapping, MutableMapping
from typing import cast, Iterator, Sequence
import logging
import operator
import itertools

import numpy as np
from scipy.sparse import csc_matrix

from qiskit_nature.exceptions import QiskitNatureError

from .polynomial_tensor import PolynomialTensor
from .sparse_label_op import SparseLabelOp

logger = logging.getLogger(__name__)


class VibrationalOp(SparseLabelOp):
    r"""N-mode vibrational operator.

    A `VibrationalOp` represents a weighted sum of vibrational creation/annihilation operator terms.
    These terms are encoded as sparse labels, strings consisting of a space-separated list of
    expressions. Each expression must look like :code:`[+-]_<mode_index>_<modal_index>`, where the
    :code:`<mode_index>` and :code:`<modal_index> are non-negative integers representing the index
    of the vibrational mode and modal, respectively, where the `+` (creation) or `-` (annihilation)
    operation is to be performed. The value of :code:`mode_index` is bound by the number of modes
    (`num_modes`) of the operator (Note: since Python indices are 0-based, the maximum value an
    index can take is given by :code:`num_modes-1`). Similarly, the value of :code:`modal_index` has
    an upper bound given by :code:`num_modals-1` for that particular mode.

    **Initialization**

    A `VibrationalOp` is initialized with a dictionary, mapping terms to their respective
    coefficients:

    .. jupyter-execute::

        from qiskit_nature.second_q.operators import VibrationalOp

        op = VibrationalOp(
            {
                "+_0_0 -_0_0": 1.0,
                "+_0_1 -_0_1": 1.0,
                "+_1_0 -_1_0": -1.0,
                "+_1_1 -_1_1": -1.0,
            },
            num_modes=2,
            num_modals=2,
        )

    By default, this way of initializing will create a full copy of the dictionary of coefficients.
    If you have very restricted memory resources available, or would like to avoid the additional
    copy, the dictionary will be stored by reference if you disable ``copy`` like so:

    .. jupyter-execute::

        some_big_data = {
            "+_0_0 -_0_0": 1.0,
            "+_0_1 -_0_1": 1.0,
            # ...
        }

        op = VibrationalOp(
            some_big_data,
            num_modes=2,
            num_modals=2,
            copy=False,
        )

    .. note::

        It is the users' responsibility, that in the above scenario, :code:`some_big_data` is not
        changed after initialization of the `VibrationalOp`, since the operator contents are not
        guaranteed to remain unaffected by such changes.

    If :code:`num_modes` is not specified then it will set by the maximum :code:`mode_index` in
    :code:`data`. If :code:`num_modals` is not provided then the maximum :code:`modal_index` per
    mode will determine the :code:`num_modals` for that mode.

    .. jupyter-execute::

        from qiskit_nature.second_q.operators import VibrationalOp

        op = VibrationalOp(
            {
                "+_0_0 -_0_0": 1.0,
                "+_0_1 -_0_1": 1.0,
                "+_1_0 -_1_0": -1.0,
                "+_1_1 -_1_1": -1.0,
            },
        )


    **Algebra**

    This class supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, operator multiplication, and adjoint.
    For example,

    Addition

    .. jupyter-execute::

      VibrationalOp({"+_1_0": 1}, num_modes=2, num_modals=2) +
      VibrationalOp({"+_0_0": 1}, num_modes=2, num_modals=2)

    Sum

    .. jupyter-execute::

      sum(VibrationalOp({label: 1}, num_modes=3) for label in ["+_0_0", "-_1_0", "+_2_0 -_2_)"])

    Scalar multiplication

    .. jupyter-execute::

      0.5 * VibrationalOp({"+_1_0": 1}, num_modes=2)

    Operator multiplication

    .. jupyter-execute::

      op1 = VibrationalOp({"+_0_0 -_1_0": 1}, num_modes=2)
      op2 = VibrationalOp({"-_0_0 +_0_0 +_1_0": 1}, num_modes=2)
      print(op1 @ op2)

    Tensor multiplication

    .. jupyter-execute::

      op = VibrationalOp({"+_0_0 -_1_0": 1}, num_modes=2)
      print(op ^ op)

    Adjoint

    .. jupyter-execute::

      VibrationalOp({"+_0_0 -_1_0": 1j}, num_modes=2).adjoint()

    In principle, you can also add :class:`VibrationalOp` and integers, but the only valid case is the
    addition of `0 + VibrationalOp`. This makes the `sum` operation from the example above possible
    and it is useful in the following scenario:

    .. code-block:: python

        vibrational_op = 0
        for i in some_iterable:
            # some processing
            vibrational_op += VibrationalOp(somedata)

    **Iteration**

    Instances of `VibrationalOp` are iterable. Iterating a VibrationalOp yields (term, coefficient)
    pairs describing the terms contained in the operator.

    Attributes:
        num_modes (int | None): the number of vibrational modes on which this operator acts. This is
            considered a lower bound, which means that mathematical operations acting on two or more
            operators will result in a new operator with the maximum number of modes of any
            of the involved operators.
        num_modals (int | Sequence[int] | None): the number of modals - described by a list of integers
            where each integer describes the number of modals in a corresponding mode; in case of
            the same number of modals in each mode it is enough to provide an integer that describes
            the number of them; the total number of modals defines a ``register_length``. This is
            considered a lower bound.
    """

    # a valid pattern consists of a single "+" or "-" operator followed by "_" and a mode index
    # followed by "_" and a modal index, possibly appearing multiple times and separated by a space
    _OPERATION_REGEX = re.compile(r"^([\+\-]_\d+_\d+\s)*[\+\-]_\d+_\d+(?!\s)$|^[\+\-]+$")

    _SIMPLIFIED_REGEX = re.compile(r"([\+\-]_\d+\s)*[\+\-]_\d+")

    def __init__(
        self,
        data: Mapping[str, complex],
        num_modes: int | None = None,
        num_modals: int | Sequence[int] | None = None,
        *,
        copy: bool = True,
        validate: bool = True,
    ) -> None:
        """
        Args:
            data: the operator data, mapping string-based keys to numerical values.
            num_modes: number of modes on which this operator acts.
            num_modals: number of modals - described by a list of integers where each integer
                describes the number of modals in a corresponding mode; in case of the same number
                of modals in each mode it is enough to provide an integer that describes the number
                of them; the total number of modals defines a ``register_length``.
            copy: when set to False the `data` will not be copied and the dictionary will be
                stored by reference rather than by value (which is the default; `copy=True`). Note,
                that this requires you to not change the contents of the dictionary after
                constructing the operator. This also implies `validate=False`. Use with care!
            validate: when set to False the `data` keys will not be validated. Note, that the
                SparseLabelOp base class, makes no assumption about the data keys, so will not
                perform any validation by itself. Only concrete subclasses are encouraged to
                implement a key validation method. Disable this setting with care!

        Raises:
            QiskitNatureError: when an invalid key is encountered during validation.
            QiskitNatureError: when the number of modes and sequence of modals do not match.
        """
        self.num_modes = num_modes

        # if num_modes is None and num_modals is a sequence, get num_modes from sequence length.
        # if num_modes is None and num_modals is an int leave num_modes unbounded.
        # if num_modes is int and num_modals is None create list of zeros of length num_modals.
        # if both are none, leave both unbounded.

        if num_modes is not None and isinstance(num_modals, int):
            num_modals = [num_modals] * num_modes

        if num_modes is not None and num_modals is None:
            num_modals = [0] * num_modes

        if num_modes is None and isinstance(num_modals, Sequence):
            num_modes = len(num_modals)

        # if (
        #     isinstance(num_modals, Sequence)
        #     and isinstance(num_modes, int)
        #     and len(num_modals) != num_modes
        # ):
        #     raise QiskitNatureError(
        #         f"Length of num_modals ({len(num_modals)}) not equal to num_modes ({num_modes})"
        #     )

        self.num_modals = num_modals

        super().__init__(data, copy=copy, validate=validate)

    # @property
    # def num_modes(self) -> int:
    #     return self._num_modes

    # @num_modes.setter
    # def num_modes(self, num_modes: int | None):
    #     self._num_modes = num_modes

    # @property
    # def num_modals(self) -> Sequence[int]:
    #     return self._num_modals

    # @num_modals.setter
    # def num_modes(self, num_modals: Sequence[int] | int | None):
    #     self._num_modes = num_modals

    @property
    def register_length(self) -> int | None:
        return sum(self.num_modals)

    def _new_instance(
        self, data: Mapping[str, complex], *, other: VibrationalOp | None = None
    ) -> VibrationalOp:
        # Not sure how to deal with int num_modals and list other_num_modals and vice versa.
        # Assume list is preferable?
        num_modes = self.num_modes
        num_modals = self.num_modals
        if other is not None:
            other_num_modes = other.num_modes
            if num_modes is None:
                num_modes = other_num_modes
            elif other_num_modes is not None:
                num_modes = max(num_modes, other_num_modes)

            other_num_modals = other.num_modals
            if num_modals is None:
                num_modals = other_num_modals
            elif other_num_modals is not None:
                if isinstance(num_modals, int) and isinstance(other_num_modals, int):
                    num_modals = max(num_modals, other_num_modals)

                if isinstance(num_modals, list) and isinstance(other_num_modals, list):

                    def pad_to_length(a, b):
                        if len(a) < len(b):
                            a, b = b, a
                        return a, b + [0] * (len(a) - len(b))

                    def elementwise_max(a, b):
                        return [max(i, j) for i, j in zip(*pad_to_length(a, b))]

                    num_modals = elementwise_max(num_modals, other_num_modals)
                    num_modes = len(num_modals)

        return self.__class__(data, copy=False, num_modes=num_modes, num_modals=num_modals)

    def _validate_keys(self, keys: Collection[str]) -> None:

        # After validating the keys, num_modals should *always* be a sequence of ints.

        super()._validate_keys(keys)

        for key in keys:
            # 0. explicitly allow the empty key
            if key == "":
                continue

            # Validate overall key structure
            if not re.fullmatch(VibrationalOp._OPERATION_REGEX, key):
                raise QiskitNatureError(f"{key} is not a valid VibrationalOp label.")

        self._validate_vibrational_indices(keys)

    def _validate_vibrational_indices(self, vibrational_labels: Collection[str]):
        num_modes = self.num_modes
        num_modals = self.num_modals

        for labels in vibrational_labels:
            coeff_labels_split = labels.split()
            for label in coeff_labels_split:
                _, mode_index_str, modal_index_str = re.split("[_]", label)
                mode_index = int(mode_index_str)
                modal_index = int(modal_index_str)

                if num_modes is None:
                    num_modes = mode_index + 1
                    if isinstance(num_modals, int):
                        num_modals = num_modes * [num_modals]

                if num_modals is None:
                    num_modals = [0] * num_modes

                if mode_index > num_modes - 1:
                    num_modals += [0] * (mode_index - num_modes + 1)
                    num_modes = mode_index + 1

                if modal_index > num_modals[mode_index] - 1:
                    num_modals[mode_index] = modal_index + 1

        if num_modes is None:
            num_modes = 0

        if num_modals is None:
            num_modals = [0]

        # TODO merge this into the loop above
        # for labels in vibrational_labels:
        #     par_num_mode_conserved_check = [0] * num_modes
        #     for label in labels.split():
        #         op, mode_index_str, modal_index_str = re.split("[_]", label)
        #         mode_index = int(mode_index_str)
        #         modal_index = int(modal_index_str)
        #         par_num_mode_conserved_check[int(mode_index)] += 1 if op == "+" else -1
        #     for index, item in enumerate(par_num_mode_conserved_check):
        #         if item != 0:
        #             logger.warning(
        #                 "Number of raising and lowering operators do not agree for mode %s in "
        #                 "label %s.",
        #                 index,
        #                 labels,
        #             )

        self.num_modes = num_modes
        self.num_modals = num_modals

    @classmethod
    def _validate_polynomial_tensor_key(cls, keys: Collection[str]) -> None:
        allowed_chars = {"+", "-"}

        for key in keys:
            if set(key) - allowed_chars:
                raise QiskitNatureError(
                    f"The key {key} is invalid. PolynomialTensor keys may only consists of `+` and "
                    "`-` characters, for them to be expandable into a VibrationalOp."
                )

    @classmethod
    def from_polynomial_tensor(cls, tensor: PolynomialTensor) -> VibrationalOp:
        cls._validate_polynomial_tensor_key(tensor.keys())

        data: dict[str, complex] = {}

        for key in tensor:
            if key == "":
                # TODO: deal with complexity
                data[""] = cast(float, tensor[key])
                continue

            label_template = " ".join(f"{op}_{{}}" for op in key)

            # PERF: this matrix unpacking is a performance bottleneck
            # we could consider using Rust in the future to improve upon this

            ndarray = cast(np.ndarray, tensor[key])
            for index in np.ndindex(*ndarray.shape):
                data[label_template.format(*index)] = ndarray[index]

            # NOTE: once the PolynomialTensor supports sparse matrices, these will need to be
            # handled separately

        return cls(data, copy=False, num_spin_orbitals=tensor.register_length).chop()

    def __repr__(self) -> str:
        data_str = f"{dict(self.items())}"

        return "VibrationalOp(" f"{data_str}, " f"num_spin_orbitals={self.num_spin_orbitals}, " ")"

    def __str__(self) -> str:
        pre = (
            "Vibrational Operator\n"
            f"number modes={self.num_modes}, number modals={self.num_modals}, "
            f"number terms={len(self)}\n"
        )
        ret = "  " + "\n+ ".join(
            [f"{coeff} * ( {label} )" if label else f"{coeff}" for label, coeff in self.items()]
        )
        return pre + ret

    def terms(self) -> Iterator[tuple[list[tuple[str, int]], complex]]:
        """Provides an iterator analogous to :meth:`items` but with the labels already split into
        pairs of operation characters and indices.

        Yields:
            A tuple with two items; the first one being a list of pairs of the form (char, int)
            where char is either `+` or `-` and the integer corresponds to the vibrational mode and
            modal index on which the operator gets applied; the second item of the returned tuple is
            the coefficient of this term.
        """
        num_modals = self.num_modals
        partial_sum_modals = [0] + list(itertools.accumulate(num_modals, operator.add))

        for label in iter(self):
            if not label:
                yield ([], self[label])
                continue
            # we hard-code the result of lbl.split("_") as follows:
            #   lbl[0] is either + or -
            #   lbl[2:] corresponds to the index
            terms = [
                self._build_register_label(lbl, partial_sum_modals) for lbl in label.split(" ")
            ]
            yield (terms, self[label])

    def _build_register_label(self, label: str, partial_sum_modals: list[int]) -> tuple[str, int]:
        op, mode_index, modal_index = re.split("[_]", label)
        index = partial_sum_modals[int(mode_index)] + int(modal_index)
        return (op, index)

    def compose(self, other: VibrationalOp, qargs=None, front: bool = False) -> VibrationalOp:
        if not isinstance(other, VibrationalOp):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'VibrationalOp' and '{type(other).__name__}'"
            )

        if front:
            return self._tensor(self, other, offset=False)
        else:
            return self._tensor(other, self, offset=False)

    def tensor(self, other: VibrationalOp) -> VibrationalOp:
        return self._tensor(self, other)

    def expand(self, other: VibrationalOp) -> VibrationalOp:
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a: VibrationalOp, b: VibrationalOp, *, offset: bool = True) -> VibrationalOp:
        shift = a.num_modes if offset else 0

        new_data: dict[str, complex] = {}
        for labels1, cf1 in a.items():
            for labels2, cf2 in b.items():
                if labels2 == "":
                    new_label = labels1
                else:
                    terms = [re.split("[*_]", lbl) for lbl in labels2.split(" ")]
                    new_label = (
                        f"{labels1} {' '.join(f'{c}_{int(i)+shift}_{j}' for c, i, j in terms)}".strip()
                    )
                if new_label in new_data:
                    new_data[new_label] += cf1 * cf2
                else:
                    new_data[new_label] = cf1 * cf2

        new_op = a._new_instance(new_data, other=b)
        if offset:
            # num_modals should always be a list after instantiation
            new_op.num_modes = a.num_modes + b.num_modes
            new_op.num_modals = a.num_modals.extend(b.num_modals)
        return new_op

    def to_matrix(self, sparse: bool | None = True) -> csc_matrix | np.ndarray:
        """Convert to a matrix representation over the full vibrational Fock space in the occupation
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
            for terms, prefactor in self.simplify().terms():
                # check if op string is the identity
                if not terms:
                    csc_data.append(prefactor)
                    csc_row.append(col_idx)
                    csc_col.append(col_idx)
                else:
                    occupations = initial_occupations.copy()
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
                        occupations[index] = not occ

                    # add data point to matrix in the correct row
                    if not mapped_to_zero:
                        row_idx = sum(int(occ) << idx for idx, occ in enumerate(occupations[::-1]))
                        csc_data.append(prefactor)
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

    def transpose(self) -> VibrationalOp:
        data = {}

        trans = "".maketrans("+-", "-+")

        for label, coeff in self.items():
            data[" ".join(lbl.translate(trans) for lbl in reversed(label.split(" ")))] = coeff

        return self._new_instance(data)

    def is_hermitian(self, *, atol: float | None = None) -> bool:
        """Checks whether the operator is hermitian.

        Args:
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``.

        Returns:
            True if the operator is hermitian up to numerical tolerance, False otherwise.
        """
        atol = self.atol if atol is None else atol
        diff = (self - self.adjoint()).simplify(atol=atol)
        return all(np.isclose(coeff, 0.0, atol=atol) for coeff in diff.values())

    def simplify(self, *, atol: float | None = None) -> VibrationalOp:
        atol = self.atol if atol is None else atol

        data = defaultdict(complex)  # type: dict[str, complex]
        # TODO: use parallel_map to make this more efficient (?)
        for label, coeff in self.items():
            label, coeff = self._simplify_label(label, coeff)
            data[label] += coeff
        simplified_data = {
            label: coeff for label, coeff in data.items() if not np.isclose(coeff, 0.0, atol=atol)
        }
        return self._new_instance(simplified_data)

    def _simplify_label(self, label: str, coeff: complex) -> tuple[str, complex]:
        bits = _BitsContainer()

        # Since Python 3.7, dictionaries are guaranteed to be insert-order preserving. We use this
        # to our advantage, to implement an ordered set, which allows us to preserve the label order
        # and only remove canceling terms.
        new_label: dict[str, None] = {}

        for lbl in label.split():
            char, mode_index, modal_index = lbl.split("_")
            idx = (int(mode_index), int(modal_index))
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
                pop_lbl = f"{'-' if char_b else '+'}_{idx[0]}_{idx[1]}"
                # 2. we perform the information update by:
                #  a) updating the coefficient sign
                new_label.pop(pop_lbl)
                #  b) updating the bits container
                bits.set_plus_or_minus(idx, not char_b, False)
                #  c) and updating the last bit to the current char
                bits.set_last(idx, char_b)

            else:
                # else, we simply set the bit of the currently applied char
                bits.set_plus_or_minus(idx, char_b, True)
                # and track it in our ordered set
                new_label[lbl] = None
                # we also update the last bit to the current char
                bits.set_last(idx, char_b)

        return " ".join(new_label), coeff


class _BitsContainer(MutableMapping):
    """A bit-storage container.

    This is a utility object used during the simplification process of a `VibrationalOp`.
    It manages access to an internal data container, which maps from integers to bytes.
    Each integer key corresponds to a vibrational mode of an operator term.
    Each value consists of four bits which encoding for the corresponding index:

        1. if a `+` has been applied
        2. if a `-` has been applied
        3. whether a `+` or `-` was applied first
        4. whether the last applied operator was a `+` or `-`
    """

    def __init__(self):
        self.data: dict[tuple[int, int], int] = {}

    def get_plus(self, index: tuple[int, int]) -> int:
        """Returns the value of the `+`-register.

        Args:
            index: the internal data key (corresponding to the vibrational mode).

        Returns:
            1 if `+` has been applied, 0 otherwise.
        """
        return self.get_bit(index, 3)

    def get_minus(self, index: tuple[int, int]) -> int:
        """Returns the value of the `-`-register.

        Args:
            index: the internal data key (corresponding to the vibrational mode).

        Returns:
            1 if `-` has been applied, 0 otherwise.
        """
        return self.get_bit(index, 2)

    def set_plus_or_minus(self, index: tuple[int, int], plus_or_minus: bool, value: bool) -> None:
        """Sets the `+`- or `-`-register of the provided index to the provided value.

        Args:
            index: the internal data key (corresponding to the vibrational mode).
            plus_or_minus: True if the `+`-register is to be set, False for the `-`-register
            value: True if the register is to be set to 1, False for 0.
        """
        if value:
            # plus is stored at index 0, but plus_or_minus is True if it is Plus
            self.set_bit(index, 3 - int(not plus_or_minus))
        else:
            self.clear_bit(index, 3 - int(not plus_or_minus))

    def get_order(self, index: tuple[int, int]) -> int:
        """Returns the value of the order-register.

        Note: the order-register is read-only and can only be set during initialization.

        Args:
            index: the internal data key (corresponding to the vibrational mode).

        Returns:
            1 if `+` was applied first, 0 if `-` was applied first.
        """
        return self.get_bit(index, 1)

    def get_last(self, index: tuple[int, int]) -> int:
        """Returns the value of the last-register.

        Args:
            index: the internal data key (corresponding to the vibrational mode).

        Returns:
            1 if `+` was applied last, 0 otherwise.
        """
        return self.get_bit(index, 0)

    def set_last(self, index: tuple[int, int], value: bool) -> None:
        """Sets the value of the last-register.

        Args:
            index: the internal data key (corresponding to the vibrational mode).
            value: True if the register is to be set to 1, False for 0.
        """
        if value:
            self.set_bit(index, 0)
        else:
            self.clear_bit(index, 0)

    def get_bit(self, index: tuple[int, int], offset: int) -> int:
        """Returns the value of a requested register.

        Args:
            index: the internal data key (corresponding to the vibrational mode).
            offset: the bit-wise offset for the bit-shift operation to obtain the desired register.

        Returns:
            1 if the register was set, 0 otherwise.
        """
        return (self.data[index] >> offset) & 1

    def set_bit(self, index: tuple[int, int], offset: int) -> None:
        """Sets the provided register to 1.

        Args:
            index: the internal data key (corresponding to the vibrational mode).
            offset: the bit-wise offset for the bit-shift operation to set the desired register.
        """
        self.data[index] = self.data[index] | (1 << offset)

    def clear_bit(self, index: tuple[int, int], offset: int) -> None:
        """Clears the provided register (to 0).

        Args:
            index: the internal data key (corresponding to the vibrational mode).
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
