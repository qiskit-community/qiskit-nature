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
from collections.abc import Collection, Mapping
from typing import Iterator, Sequence
import logging
import operator
import itertools

import numpy as np

from qiskit_nature.exceptions import QiskitNatureError

from ._bits_container import _BitsContainer
from .polynomial_tensor import PolynomialTensor
from .sparse_label_op import SparseLabelOp

logger = logging.getLogger(__name__)

def build_dual_index(num_modals: Sequence[int], index: int) -> str:
    r"""Convert a single expanded index into a dual index.

    Args:
        num_modals: The number of modals - described by a list of integers where each integer
            describes the number of modals in the corresponding mode; the total number of modals
            defines a ``register_length``.
        index: The expanded (register) index.

    Returns

    Raises:
        ValueError: If the index is greater than the sum of num_modals.
    """

    for mode_index, num_modals_per_mode in enumerate(num_modals):
        if index < num_modals_per_mode:
            return f"{mode_index}_{index}"
        else:
            index -= num_modals_per_mode

    raise ValueError("Invalid index: index > sum(num_modals) - 1.")


class VibrationalOp(SparseLabelOp):
    r"""N-mode vibrational operator.

    A `VibrationalOp` represents a weighted sum of vibrational creation/annihilation operator terms.
    These terms are encoded as sparse labels, strings consisting of a space-separated list of
    expressions. Each expression must look like :code:`[+-]_<mode_index>_<modal_index>`, where the
    :code:`<mode_index>` and :code:`<modal_index> are non-negative integers representing the index
    of the vibrational mode and modal, respectively, where the `+` (creation) or `-` (annihilation)
    operation is to be performed.

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
            num_modals=[2, 2]
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
            num_modals=[2, 2],
            copy=False,
        )

    .. note::

        It is the users' responsibility, that in the above scenario, :code:`some_big_data` is not
        changed after initialization of the `VibrationalOp`, since the operator contents are not
        guaranteed to remain unaffected by such changes.

    If :code:`num_modals` is not provided then the maximum :code:`modal_index` per
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

      VibrationalOp({"+_1_0": 1}, num_modals=[2, 2]) +
      VibrationalOp({"+_0_0": 1}, num_modals=[2, 2])

    Sum

    .. jupyter-execute::

      sum(VibrationalOp({label: 1}, num_modals=[1, 1, 1]) for label in ["+_0_0", "-_1_0", "+_2_0 -_2_0"])

    Scalar multiplication

    .. jupyter-execute::

      0.5 * VibrationalOp({"+_1_0": 1}, num_modals=[1, 1])

    Operator multiplication

    .. jupyter-execute::

      op1 = VibrationalOp({"+_0_0 -_1_0": 1}, num_modals=[1, 1])
      op2 = VibrationalOp({"-_0_0 +_0_0 +_1_0": 1}, num_modals=[1, 1])
      print(op1 @ op2)

    Tensor multiplication

    .. jupyter-execute::

      op = VibrationalOp({"+_0_0 -_1_0": 1}, num_modals=[1, 1])
      print(op ^ op)

    Adjoint

    .. jupyter-execute::

      VibrationalOp({"+_0_0 -_1_0": 1j}, num_modals=[1, 1]).adjoint()

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
        num_modals (Sequence[int] | None): the number of modals - described by a list of integers
            where each integer describes the number of modals in a corresponding mode; the total
            number of modals defines a ``register_length``. This is considered a lower bound.
    """

    # a valid pattern consists of a single "+" or "-" operator followed by "_" and a mode index
    # followed by "_" and a modal index, possibly appearing multiple times and separated by a space
    _OPERATION_REGEX = re.compile(r"([\+\-]_\d+_\d+\s)*[\+\-]_\d+_\d+(?!\s)")

    def __init__(
        self,
        data: Mapping[str, complex],
        num_modals: Sequence[int] | None = None,
        *,
        copy: bool = True,
        validate: bool = True,
    ) -> None:
        """
        Args:
            data: the operator data, mapping string-based keys to numerical values.
            num_modals: number of modals - described by a list of integers where each integer
                describes the number of modals in the corresponding mode; the total number of modals
                defines a ``register_length``.
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
        """
        self.num_modals = num_modals
        super().__init__(data, copy=copy, validate=validate)

    @property
    def num_modals(self) -> Sequence[int]:
        """The number of modals.
        
        Described by a list of integers where each integer describes the number of modals in the
        corresponding mode; the total number of modals defines a ``register_length``.
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
        self, data: Mapping[str, complex], *, other: VibrationalOp | None = None
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
        num_modals = self.num_modals

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
        ...

    @classmethod
    def from_polynomial_tensor(cls, tensor: PolynomialTensor) -> VibrationalOp:
        ...

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
            terms = [
                self._build_register_label(lbl, partial_sum_modals) for lbl in label.split(" ")
            ]
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

        new_data: dict[str, complex] = {}
        for labels1, cf1 in a.items():
            for labels2, cf2 in b.items():
                if labels2 == "":
                    new_label = labels1
                else:
                    terms = [lbl.split("_") for lbl in labels2.split(" ")]
                    new_label = f"{labels1} {' '.join(f'{c}_{int(i)+shift}_{j}' for c, i, j in terms)}".strip()
                if new_label in new_data:
                    new_data[new_label] += cf1 * cf2
                else:
                    new_data[new_label] = cf1 * cf2

        new_op = a._new_instance(new_data, other=b)
        if offset:
            new_op.num_modals = a.num_modals.extend(b.num_modals)
        return new_op

    def transpose(self) -> VibrationalOp:
        data = {}

        trans = "".maketrans("+-", "-+")

        for label, coeff in self.items():
            data[" ".join(lbl.translate(trans) for lbl in reversed(label.split(" ")))] = coeff

        return self._new_instance(data)

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
        bits = _BitsContainer[tuple[int, int]]()

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
                #  a) popping the key we just canceled outn
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
