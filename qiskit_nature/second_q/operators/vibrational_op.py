# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
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
from collections.abc import Collection, Mapping
from typing import Iterator, Sequence, Tuple, cast
import logging
import operator
import itertools

import numpy as np

from qiskit_nature.exceptions import QiskitNatureError

from ._bits_container import _BitsContainer
from .polynomial_tensor import PolynomialTensor
from .sparse_label_op import _TCoeff, SparseLabelOp, _to_number
from .tensor import Tensor

logger = logging.getLogger(__name__)


class VibrationalOp(SparseLabelOp):
    r"""N-mode vibrational operator.

    A ``VibrationalOp`` represents a weighted sum of vibrational creation/annihilation operator terms.
    These terms are encoded as sparse labels, strings consisting of a space-separated list of
    expressions. Each expression must look like :code:`[+-]_<mode_index>_<modal_index>`, where the
    :code:`<mode_index>` and :code:`<modal_index>` are non-negative integers representing the index
    of the vibrational mode and modal, respectively, where the ``+`` (creation) or ``-`` (annihilation)
    operation is to be performed.

    **Initialization**

    A ``VibrationalOp`` is initialized with a dictionary, mapping terms to their respective
    coefficients:

    .. code-block:: python

        from qiskit_nature.second_q.operators import VibrationalOp

        op = VibrationalOp(
            {
                "+_0_0 -_0_0": 1.0,
                "+_0_1 -_0_1": 1.0,
                "+_1_0 -_1_0": -1.0,
                "+_1_1 -_1_1": -1.0,
            },
            num_modals=[2, 2]
        )

    By default, this way of initializing will create a full copy of the dictionary of coefficients.
    If you have very restricted memory resources available, or would like to avoid the additional
    copy, the dictionary will be stored by reference if you disable ``copy`` like so:

    .. code-block:: python

        some_big_data = {
            "+_0_0 -_0_0": 1.0,
            "+_0_1 -_0_1": 1.0,
            # ...
        }

        op = VibrationalOp(
            some_big_data,
            num_modals=[2, 2],
            copy=False,
        )

    .. note::

        It is the users' responsibility, that in the above scenario, :code:`some_big_data` is not
        changed after initialization of the ``VibrationalOp``, since the operator contents are not
        guaranteed to remain unaffected by such changes.

    If :code:`num_modals` is not provided then the maximum :code:`modal_index` per
    mode will determine the :code:`num_modals` for that mode.

    .. code-block:: python

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

    .. code-block:: python

      VibrationalOp({"+_1_0": 1}, num_modals=[2, 2]) + VibrationalOp({"+_0_0": 1}, num_modals=[2, 2])

    Sum

    .. code-block:: python

      sum(VibrationalOp({label: 1}, num_modals=[1, 1, 1]) for label in ["+_0_0", "-_1_0", "+_2_0 -_2_0"])

    Scalar multiplication

    .. code-block:: python

      0.5 * VibrationalOp({"+_1_0": 1}, num_modals=[1, 1])

    Operator multiplication

    .. code-block:: python

      op1 = VibrationalOp({"+_0_0 -_1_0": 1}, num_modals=[1, 1])
      op2 = VibrationalOp({"-_0_0 +_0_0 +_1_0": 1}, num_modals=[1, 1])
      print(op1 @ op2)

    Tensor multiplication

    .. code-block:: python

      op = VibrationalOp({"+_0_0 -_1_0": 1}, num_modals=[1, 1])
      print(op ^ op)

    Adjoint

    .. code-block:: python

      VibrationalOp({"+_0_0 -_1_0": 1j}, num_modals=[1, 1]).adjoint()

    **Iteration**

    Instances of ``VibrationalOp`` are iterable. Iterating a ``VibrationalOp`` yields (term, coefficient)
    pairs describing the terms contained in the operator.

    .. note::

        A VibrationalOp can contain :class:`qiskit.circuit.ParameterExpression` objects as
        coefficients.
    """

    # a valid pattern consists of a single "+" or "-" operator followed by "_" and a mode index
    # followed by "_" and a modal index, possibly appearing multiple times and separated by a space
    _OPERATION_REGEX = re.compile(r"([\+\-]_\d+_\d+\s)*[\+\-]_\d+_\d+(?!\s)")

    def __init__(
        self,
        data: Mapping[str, _TCoeff],
        num_modals: Sequence[int] | None = None,
        *,
        copy: bool = True,
        validate: bool = True,
    ) -> None:
        """
        Args:
            data: the operator data, mapping string-based keys to numerical values.
            num_modals: number of modals - described by a sequence of integers where each integer
                describes the number of modals in the corresponding mode; the total number of modals
                defines a ``register_length``.
            copy: when set to False the `data` will not be copied and the dictionary will be
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
        self.num_modals = num_modals
        super().__init__(data, copy=copy, validate=validate)

    @property
    def num_modals(self) -> Sequence[int]:
        """The number of modals for each mode on which this operator acts.

        This is an optional sequence of integers which are considered lower bounds. That means that
        mathematical operations acting on two or more operators will result in a new operator with
        the maximum number of modals for each mode involved in any of the operators.
        """
        # to ensure future flexibility, the type here is Sequence. However, the current
        # implementation ensures it will always be a list.
        return self._num_modals

    @num_modals.setter
    def num_modals(self, num_modals: Sequence[int] | None):
        self._num_modals = list(num_modals) if num_modals is not None else []

    @property
    def register_length(self) -> int | None:
        return sum(self.num_modals) if self.num_modals is not None else None

    def _new_instance(
        self, data: Mapping[str, _TCoeff], *, other: VibrationalOp | None = None
    ) -> VibrationalOp:
        num_modals = self.num_modals
        if other is not None:
            other_num_modals = other.num_modals

            def pad_to_length(a, b):
                if len(a) < len(b):
                    a, b = b, a
                return a, b + [0] * (len(a) - len(b))

            def elementwise_max(a, b):
                return [max(i, j) for i, j in zip(*pad_to_length(a, b))]

            num_modals = elementwise_max(num_modals, other_num_modals)

        return self.__class__(data, copy=False, num_modals=num_modals)

    def _validate_keys(self, keys: Collection[str]) -> None:
        super()._validate_keys(keys)
        num_modals = list(self.num_modals)

        for key in keys:
            # 0. explicitly allow the empty key
            if key == "":
                continue

            # Validate overall key structure
            if not re.fullmatch(VibrationalOp._OPERATION_REGEX, key):
                raise QiskitNatureError(f"{key} is not a valid VibrationalOp label.")

            coeff_labels_split = key.split()
            for label in coeff_labels_split:
                _, mode_index_str, modal_index_str = label.split("_")
                mode_index = int(mode_index_str)
                modal_index = int(modal_index_str)

                if mode_index + 1 > len(num_modals):
                    num_modals += [0] * (mode_index + 1 - len(num_modals))

                if modal_index > num_modals[mode_index] - 1:
                    num_modals[mode_index] = modal_index + 1

        self.num_modals = num_modals

    @classmethod
    def _validate_polynomial_tensor_key(cls, keys: Collection[str]) -> None:
        allowed = re.compile(r"(_\+\-)*")

        for key in keys:
            if not re.fullmatch(allowed, key):
                raise QiskitNatureError(
                    f"The key '{key}' is invalid. PolynomialTensor keys must be multiples of the "
                    "'_+-' character sequence, for them to be expandable into a VibrationalOp."
                )

    @classmethod
    def from_polynomial_tensor(cls, tensor: PolynomialTensor) -> VibrationalOp:
        cls._validate_polynomial_tensor_key(tensor.keys())

        data: dict[str, _TCoeff] = {}

        def _reshape_index(index):
            new_index = []
            for idx in range(0, len(index), 3):
                new_index.extend([index[idx], index[idx + 1], index[idx], index[idx + 2]])
            return new_index

        for key in tensor:
            if key == "":
                # TODO: deal with complexity
                data[""] = cast(float, tensor[key])
                continue

            mat = tensor[key]

            if not isinstance(mat, Tensor):
                # TODO: this case is to be removed once qiskit_nature.settings.tensor_unwrapping is
                # deprecated and the PolynomialTensor item is guaranteed to be of type Tensor
                mat = Tensor(mat)

            label_template = mat.label_template.format(*key.replace("_", ""))

            for value, index in mat.coord_iter():
                data[label_template.format(*_reshape_index(index))] = value

        return cls(data)

    def __repr__(self) -> str:
        data_str = f"{dict(self.items())}"

        return "VibrationalOp(" f"{data_str}, " f"num_modals={self.num_modals}, " ")"

    def __str__(self) -> str:
        pre = (
            "Vibrational Operator\n"
            f"number modes={len(self.num_modals)}, number modals={self.num_modals}, "
            f"number terms={len(self)}\n"
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
            where char is either ``+`` or ``-`` and the integer corresponds to the vibrational mode and
            modal index on which the operator gets applied; the second item of the returned tuple is
            the coefficient of this term.
        """
        num_modals = self.num_modals
        partial_sum_modals = [0] + list(itertools.accumulate(num_modals, operator.add))

        for label in iter(self):
            if not label:
                yield ([], self[label])
                continue
            terms = [self._build_register_label(lbl, partial_sum_modals) for lbl in label.split()]
            yield (terms, self[label])

    def _build_register_label(self, label: str, partial_sum_modals: list[int]) -> tuple[str, int]:
        op, mode_index, modal_index = label.split("_")
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
        shift = len(a.num_modals) if offset else 0

        new_data: dict[str, _TCoeff] = {}
        for a_labels, a_coeff in a.items():
            for b_labels, b_coeff in b.items():
                if b_labels == "":
                    new_label = a_labels
                else:
                    b_terms = [lbl.split("_") for lbl in b_labels.split()]
                    new_b_label = " ".join(f"{op}_{int(i)+shift}_{j}" for op, i, j in b_terms)
                    new_label = f"{a_labels} {new_b_label}".strip()

                if new_label in new_data:
                    new_data[new_label] += a_coeff * b_coeff
                else:
                    new_data[new_label] = a_coeff * b_coeff

        new_op = a._new_instance(new_data, other=b)
        if offset:
            new_op.num_modals = [*a.num_modals, *b.num_modals]
        return new_op

    def transpose(self) -> VibrationalOp:
        data = {}

        trans = "".maketrans("+-", "-+")

        for label, coeff in self.items():
            data[" ".join(lbl.translate(trans) for lbl in reversed(label.split()))] = coeff

        return self._new_instance(data)

    def normal_order(self) -> VibrationalOp:
        """Convert to the equivalent operator in normal order.

        The normal order for this operator is defined as follows:
        - creation (``+``) operations are applied before annihilation (``-``) ones
        - operators are ordered by index within each of the operator type groups

        Returns a new operator (the original operator is not modified).

        .. note::

            The operations encoded by a ``VibrationalOp`` are fully commutative, which means that
            re-ordering of individual terms does **not** result in a phase shift.

        Returns:
            The normal ordered operator.
        """
        ordered_op = VibrationalOp.zero()

        for label, coeff in self.items():
            terms = []
            for lbl in label.split():
                char, mode, modal = lbl.split("_")
                terms.append((char, int(mode), int(modal)))
            ordered_op += self._normal_order(terms, coeff)

        # after successful normal ordering, we remove all zero coefficients
        return self._new_instance(
            {
                label: coeff
                for label, coeff in ordered_op.items()
                if not np.isclose(_to_number(coeff), 0.0, atol=self.atol)
            }
        )

    def _normal_order(self, terms: list[tuple[str, int, int]], coeff: _TCoeff) -> VibrationalOp:
        if not terms:
            return self._new_instance({"": coeff})

        ordered_op = VibrationalOp.zero()

        # perform insertion sorting
        for i in range(1, len(terms)):
            for j in range(i, 0, -1):
                right = terms[j]
                left = terms[j - 1]

                if right[0] == "+" and left[0] == "-":
                    # swap terms where an annihilation operator is left of a creation operator
                    terms[j - 1] = right
                    terms[j] = left

                    if right[1] == left[1] and right[2] == left[2]:
                        # if their indices are identical, we incur an additional term because of:
                        # a_i a_i^\dagger = 1 + a_i^\dagger a_i
                        new_terms = terms[: (j - 1)] + terms[(j + 1) :]
                        # we can do so by recursion on this method
                        ordered_op += self._normal_order(new_terms, coeff)

                elif right[0] == left[0]:
                    # when we have identical neighboring operators, differentiate two cases:

                    # on identical index, this is an invalid operation which evaluates to
                    # zero: e.g. +_0_0 +_0_0 = 0
                    if right[1] == left[1] and right[2] == left[2]:
                        # thus, we bail on this recursion call
                        return ordered_op

                    # otherwise, if the left index is higher than the right one, swap the terms
                    elif left[1] > right[1] or (left[1] == right[1] and left[2] > right[2]):
                        terms[j - 1] = right
                        terms[j] = left

        new_label = " ".join(f"{term[0]}_{term[1]}_{term[2]}" for term in terms)
        ordered_op += self._new_instance({new_label: coeff})
        return ordered_op

    def index_order(self) -> VibrationalOp:
        """Convert to the equivalent operator with the terms of each label ordered by index.

        Returns a new operator (the original operator is not modified).

        .. note::

            You can use this method to achieve the most aggressive simplification of an operator
            without changing the operation order per index. :meth:`simplify` does *not* reorder the
            terms and, thus, cannot deduce ``-_0_0 +_1_0`` and ``+_1_0 -_0_0 +_0_0 -_0_0`` to be
            identical labels. Calling this method will reorder the latter label to
            ``-_0_0 +_0_0 -_0_0 +_1_0``, after which :meth:`simplify` will be able to correctly
            collapse these two labels into one.

        Returns:
            The index ordered operator.
        """
        data = defaultdict(complex)  # type: dict[str, _TCoeff]
        for label, coeff in self.items():
            terms = []
            for lbl in label.split():
                char, mode, modal = lbl.split("_")
                terms.append((char, int(mode), int(modal)))
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

    def _index_order(
        self, terms: list[tuple[str, int, int]], coeff: _TCoeff
    ) -> tuple[str, _TCoeff]:
        if not terms:
            return "", coeff

        # perform insertion sorting
        for i in range(1, len(terms)):
            for j in range(i, 0, -1):
                right = terms[j]
                left = terms[j - 1]

                if left[1] > right[1] or (left[1] == right[1] and left[2] > right[2]):
                    terms[j - 1] = right
                    terms[j] = left

        new_label = " ".join(f"{term[0]}_{term[1]}_{term[2]}" for term in terms)
        return new_label, coeff

    def simplify(self, atol: float | None = None) -> VibrationalOp:
        atol = self.atol if atol is None else atol

        data = defaultdict(complex)  # type: dict[str, _TCoeff]
        # TODO: use parallel_map to make this more efficient (?)
        for label, coeff in self.items():
            label, coeff = self._simplify_label(label, coeff)
            data[label] += coeff
        simplified_data = {
            label: coeff
            for label, coeff in data.items()
            if not np.isclose(_to_number(coeff), 0.0, atol=self.atol)
        }
        return self._new_instance(simplified_data)

    def _simplify_label(self, label: str, coeff: _TCoeff) -> tuple[str, _TCoeff]:
        bits = _BitsContainer[Tuple[int, int]]()

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
                #  a) popping the key we just canceled out
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

    @staticmethod
    def build_dual_index(num_modals: Sequence[int], index: int) -> str:
        r"""Convert a single expanded index into a dual mode and modal index string.

        Args:
            num_modals: The number of modals - described by a list of integers where each integer
                describes the number of modals in the corresponding mode; the total number of modals
                defines the ``register_length``.
            index: The expanded (register) index.

        Returns:
            The dual index label.

        Raises:
            ValueError: If the index is greater than the sum of num_modals.
        """

        for mode_index, num_modals_per_mode in enumerate(num_modals):
            if index < num_modals_per_mode:
                return f"{mode_index}_{index}"
            else:
                index -= num_modals_per_mode

        raise ValueError("Invalid index: index > sum(num_modals) - 1.")
