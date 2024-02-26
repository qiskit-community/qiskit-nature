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

"""The Majorana-particle Operator."""


from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Iterator, Sequence

import numpy as np

from qiskit_nature.exceptions import QiskitNatureError

from .polynomial_tensor import PolynomialTensor
from .sparse_label_op import _TCoeff, SparseLabelOp, _to_number
from .fermionic_op import FermionicOp


class MajoranaOp(SparseLabelOp):
    r"""N-mode Majorana operator.

    A ``MajoranaOp`` represents a weighted sum of Majorana fermion operator terms.
    These terms are encoded as sparse labels, which are strings consisting of a space-separated list
    of expressions. Each expression must look like :code:`_<index>`, where the :code:`<index>` is a
    non-negative integer representing the index of the mode on which the Majorana operator is
    applied. The maximum value of :code:`index` is bound by ``num_modes``. Note that, when
    converting from a ``FermionicOp`` there are two modes per spin orbital, i.e. ``num_modes`` is
    :code:`2 * FermionicOp.num_spin_orbitals - 1`

    **Initialization**

    A ``MajoranaOp`` is initialized with a dictionary, mapping terms to their respective
    coefficients:

    .. code-block:: python

        from qiskit_nature.second_q.operators import MajoranaOp

        op = MajoranaOp(
            {
                "_0 _1": .25j,
                "_1 _0": -.25j,
                "_2 _3": -.25j,
                "_3 _2": .25j,
            },
            num_modes=4,
        )

    By default, this way of initializing will create a full copy of the dictionary of coefficients.
    If you have very restricted memory resources available, or would like to avoid the additional
    copy, the dictionary will be stored by reference if you disable ``copy`` like so:

    .. code-block:: python

        some_big_data = {
            "_0 _1": .25j,
            "_1 _0": -.25j,
            # ...
        }

        op = MajoranaOp(
            some_big_data,
            num_modes=4,
            copy=False,
        )


    .. note::

        It is the users' responsibility, that in the above scenario, :code:`some_big_data` is not
        changed after initialization of the ``MajoranaOp``, since the operator contents are not
        guaranteed to remain unaffected by such changes.

    **Construction from Fermionic operator**

    As an alternative to the manual construction above, a more convenient way of initializing a
    `MajoranaOp` is, to construct it from an existing `FermionicOp`:

    .. code-block:: python

        from qiskit_nature.second_q.operators import FermionicOp, MajoranaOp
        f_op = FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=2)
        m_op = MajoranaOp.from_fermionic_op(f_op)

    Note that each ``FerminonicOp``-term consisting of :math:`n` expressions will result in a
    ``MajoranaOp``-term consisting of :math:`2^n` expressions. The conversion uses the convention
    that

    .. math::

        a_i = \frac{1}{2}(\gamma_{2i} + i \gamma_{2i+1}), \quad
        a_i^\dagger = \frac{1}{2}(\gamma_{2i} - i \gamma_{2i+1}) \,,

    where :math:`a_i` and :math:`a_i^\dagger` are the Fermionic annihilation and creation operators
    and :math:`\gamma_i` the Majorana operators.

    **Construction from a ``PolynomialTensor``**

    Using the :meth:`from_polynomial_tensor` constructor method, a ``MajoranaOp`` can be constructed
    from a :class:`~.PolynomialTensor`. In this case, the underscore character :code:`_` is the only
    allowed character in the keys of the ``PolynomialTensor``.
    For example,

    .. code-block:: python

        p_t = PolynomialTensor(
            {
                "_": np.arange(1, 3),
                "__": np.arange(1, 5).reshape((2, 2)),
            }
        )
        op = MajoranaOp.from_polynomial_tensor(p_t)

        # op is then
        MajoranaOp({'_0': 1, '_1': 2, '_0 _0': 1, '_0 _1': 2, '_1 _0': 3, '_1 _1': 4}, num_modes=2)

    **Algebra**

    This class supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, operator multiplication, and adjoint.
    For example,

    Addition

    .. code-block:: python

      MajoranaOp({"_1": 1}, num_modes=2) + MajoranaOp({"_0": 1}, num_modes=2)

    Sum

    .. code-block:: python

      sum(MajoranaOp({label: 1}, num_modes=4) for label in ["_0", "_1", "_2 _3"])

    Scalar multiplication

    .. code-block:: python

      0.5 * MajoranaOp({"_1": 1}, num_modes=2)

    Operator multiplication

    .. code-block:: python

      op1 = MajoranaOp({"_0 _1": 1}, num_modes=3)
      op2 = MajoranaOp({"_0 _1 _2": 1}, num_modes=3)
      print(op1 @ op2)

    Tensor multiplication

    .. code-block:: python

      op = MajoranaOp({"_0 _1": 1}, num_modes=2)
      print(op ^ op)

    Adjoint

    .. code-block:: python

      MajoranaOp({"_0 _1": 1j}, num_modes=2).adjoint()

    .. note::

        Since Majorana operators are self-adjoined, the adjoint of a ``MajoranaOp`` is the original
        operator with all strings reversed, e.g. :code:`"_0 _1"` becomes :code:`"_1 _0"` in the
        example above, and coefficients become complex conjugated.

    **Iteration**

    Instances of ``MajoranaOp`` are iterable. Iterating a ``MajoranaOp`` yields
    ``(term, coefficient)`` pairs describing the terms contained in the operator.

    Attributes:
        num_modes (int | None): the number of modes on which this operator acts.
            This is considered a lower bound, which means that mathematical operations acting on two
            or more operators will result in a new operator with the maximum number of modes of any
            of the involved operators.
            When converting from a ``FermionicOp``, this is twice the number of spin orbitals.

    .. note::

        ``MajoranaOp`` can contain :class:`qiskit.circuit.ParameterExpression` objects as
        coefficients. However, a ``MajoranaOp`` containing parameters does not support the following
        methods:

        - ``is_hermitian``
    """

    _OPERATION_REGEX = re.compile(r"(_\d+\s)*_\d+")

    def __init__(
        self,
        data: Mapping[str, _TCoeff],
        num_modes: int | None = None,
        *,
        copy: bool = True,
        validate: bool = True,
    ) -> None:
        """
        Args:
            data: the operator data, mapping string-based keys to numerical values.
            num_modes: the number of modes on which this operator acts.
            copy: when set to False the ``data`` will not be copied and the dictionary will be
                stored by reference rather than by value (which is the default; ``copy=True``).
                Note, that this requires you to not change the contents of the dictionary after
                constructing the operator. This also implies ``validate=False``. Use with care!
            validate: when set to False the ``data`` keys will not be validated. Note, that the
                SparseLabelOp base class, makes no assumption about the data keys, so will not
                perform any validation by itself. Only concrete subclasses are encouraged to
                implement a key validation method. Disable this setting with care!

        Raises:
            QiskitNatureError: when an invalid key is encountered during validation.
        """
        self.num_modes = num_modes
        # if num_modes is None, it is set during _validate_keys
        super().__init__(data, copy=copy, validate=validate)

    @property
    def register_length(self) -> int:
        if self.num_modes is None:
            max_index = max(int(term[1:]) for key in self._data for term in key.split())
            return max_index + 1
        return self.num_modes

    def _new_instance(
        self, data: Mapping[str, _TCoeff], *, other: MajoranaOp | None = None
    ) -> MajoranaOp:
        num_modes = self.num_modes
        if other is not None:
            other_num_modes = other.num_modes
            if num_modes is None:
                num_modes = other_num_modes
            elif other_num_modes is not None:
                num_modes = max(num_modes, other_num_modes)

        return self.__class__(data, copy=False, num_modes=num_modes)

    def _validate_keys(self, keys: Collection[str]) -> None:
        super()._validate_keys(keys)

        num_modes = self.num_modes

        max_index = -1

        for key in keys:
            # 0. explicitly allow the empty key
            if key == "":
                continue

            # 1. validate overall key structure
            if not re.fullmatch(MajoranaOp._OPERATION_REGEX, key):
                raise QiskitNatureError(f"{key} is not a valid MajoranaOp label.")

            # 2. validate all indices against register length
            for term in key.split():
                index = int(term[1:])
                if num_modes is None:
                    max_index = max(max_index, index)
                elif index >= num_modes:
                    raise QiskitNatureError(
                        f"The index, {index}, from the label, {key}, exceeds the number of "
                        f"modes, {num_modes}."
                    )

        if num_modes is None:
            self.num_modes = max_index + 1

    @classmethod
    def _validate_polynomial_tensor_key(cls, keys: Collection[str]) -> None:
        allowed_chars = {"_"}

        for key in keys:
            if set(key) - allowed_chars:
                raise QiskitNatureError(
                    f"The key {key} is invalid. PolynomialTensor keys may only consists of `_` "
                    "characters, for them to be expandable into a MajoranaOp."
                )

    @classmethod
    def from_polynomial_tensor(cls, tensor: PolynomialTensor) -> MajoranaOp:
        cls._validate_polynomial_tensor_key(tensor.keys())

        data: dict[str, _TCoeff] = {}

        for key in tensor:
            if key == "":
                data[""] = tensor[key].item()
                continue

            mat = tensor[key]

            empty_string_key = [""] * len(key)  # label format for Majorana is just '_<index>'
            label_template = mat.label_template.format(*empty_string_key)

            for value, index in mat.coord_iter():
                data[label_template.format(*index)] = value

        num_modes = tensor.register_length
        return cls(data, copy=False, num_modes=num_modes).chop()

    def __repr__(self) -> str:
        data_str = f"{dict(self.items())}"

        return "MajoranaOp(" f"{data_str}, " f"num_modes={self.num_modes}, " ")"

    def __str__(self) -> str:
        pre = "Majorana Operator\n" f"number modes={self.num_modes}, number terms={len(self)}\n"
        ret = "  " + "\n+ ".join(
            [f"{coeff} * ( {label} )" if label else f"{coeff}" for label, coeff in self.items()]
        )
        return pre + ret

    def terms(self) -> Iterator[tuple[list[tuple[str, int]], _TCoeff]]:
        """Provides an iterator analogous to :meth:`items` but with the labels already split into
        pairs of operation characters and indices.

        Yields:
            A tuple with two items; the first one being a list of pairs of the form ('', int)
            where the empty string is for compatibility with other :class:`SparseLabelOp` and
            the integer corresponds to the mode index on which the operator gets applied; the second
            item of the returned tuple is the coefficient of this term.
        """
        for label in iter(self):
            if not label:
                yield ([], self[label])
                continue
            #   label.split() will return lbl = '_<index>' for each term
            #   lbl[1:] corresponds to the index
            terms = [("", int(lbl[1:])) for lbl in label.split()]
            yield (terms, self[label])

    @classmethod
    def from_terms(cls, terms: Sequence[tuple[list[tuple[str, int]], _TCoeff]]) -> MajoranaOp:
        data = {" ".join(f"_{index}" for _, index in label): value for label, value in terms}
        return cls(data)

    @classmethod
    def from_fermionic_op(cls, op: FermionicOp, *, simplify: bool = True) -> MajoranaOp:
        """Constructs the operator from a :class:`~.FermionicOp`.

        Args:
            op: the :class:`~.FermionicOp` to convert.
            simplify: whether to index order and simplify the resulting operator.

        Returns:
            The converted :class:`~.MajoranaOp`.
        """
        data = defaultdict(complex)  # type: dict[str, _TCoeff]
        for label, coeff in op._data.items():
            terms = label.split()
            for i in range(2 ** len(terms)):
                majorana_label = ""
                coeff_power = 0
                for j, term in enumerate(terms):
                    if majorana_label:
                        majorana_label += " "
                    odd_index = (i >> j) & 1
                    index = 2 * int(term[2:]) + odd_index
                    if odd_index:
                        if term[0] == "-":
                            coeff_power += 1
                        else:
                            coeff_power += 3
                    majorana_label += f"_{index}"
                new_coeff = 1j**coeff_power * coeff / (2 ** len(terms))
                if simplify:
                    trms = next(trm for trm, _ in MajoranaOp({majorana_label: new_coeff}).terms())
                    majorana_label, new_coeff = FermionicOp._index_order(trms, new_coeff)
                    majorana_label, new_coeff = cls._simplify_label(majorana_label, new_coeff)
                data[majorana_label] += new_coeff
        return cls(data, num_modes=2 * op.num_spin_orbitals)

    def _permute_term(
        self, term: list[tuple[str, int]], permutation: Sequence[int]
    ) -> list[tuple[str, int]]:
        return [(action, permutation[index]) for action, index in term]

    def compose(self, other: MajoranaOp, qargs=None, front: bool = False) -> MajoranaOp:
        if not isinstance(other, MajoranaOp):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'MajoranaOp' and '{type(other).__name__}'"
            )

        if front:
            return self._tensor(self, other, offset=False)
        else:
            return self._tensor(other, self, offset=False)

    def tensor(self, other: MajoranaOp) -> MajoranaOp:
        return self._tensor(self, other)

    def expand(self, other: MajoranaOp) -> MajoranaOp:
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a: MajoranaOp, b: MajoranaOp, *, offset: bool = True) -> MajoranaOp:
        shift = a.num_modes if offset else 0

        new_data: dict[str, _TCoeff] = {}
        for label1, cf1 in a.items():
            for terms2, cf2 in b.terms():
                new_label = f"{label1} {' '.join(f'_{i+shift}' for _, i in terms2)}".strip()
                if new_label in new_data:
                    new_data[new_label] += cf1 * cf2
                else:
                    new_data[new_label] = cf1 * cf2

        new_op = a._new_instance(new_data, other=b)
        if offset:
            new_op.num_modes = a.num_modes + b.num_modes
        return new_op

    def transpose(self) -> MajoranaOp:
        data = {}

        for label, coeff in self.items():
            data[" ".join(lbl for lbl in reversed(label.split()))] = coeff

        return self._new_instance(data)

    def index_order(self) -> MajoranaOp:
        """Convert to the equivalent operator with the terms of each label ordered by index.

        Returns a new operator (the original operator is not modified).

        .. note::

            You can use this method to achieve the most aggressive simplification.
            :meth:`simplify` does *not* reorder the terms. For instance, using only :meth:`simplify`
            will reduce ``_2 _0 _1 _0 _0`` to ``_2 _0 _1`` but cannot deduce this label to be
            identical to ``_0 _1 _2``.
            Calling this method will reorder the former label to
            ``_0 _0 _0 _1 _2``, after which :meth:`simplify` will be able to correctly
            collapse these two labels into one.

        Returns:
            The index ordered operator.
        """
        data = defaultdict(complex)  # type: dict[str, _TCoeff]
        for terms, coeff in self.terms():
            # index ordering is identical to FermionicOp, hence we call classmethod there:
            label, coeff = FermionicOp._index_order(terms, coeff)
            data[label] += coeff

        # after successful index ordering, we remove all zero coefficients
        return self._new_instance(
            {
                label: coeff
                for label, coeff in data.items()
                if not np.isclose(_to_number(coeff), 0.0, atol=self.atol)
            }
        )

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
        diff = (self - self.adjoint()).simplify(atol=atol)
        return all(np.isclose(coeff, 0.0, atol=atol) for coeff in diff.values())

    def simplify(self, atol: float | None = None) -> MajoranaOp:
        atol = self.atol if atol is None else atol

        data = defaultdict(complex)  # type: dict[str, _TCoeff]
        # TODO: use parallel_map to make this more efficient (?) (see FermionicOp)
        for label, coeff in self.items():
            label, coeff = self._simplify_label(label, coeff)
            data[label] += coeff
        simplified_data = {
            label: coeff
            for label, coeff in data.items()
            if not np.isclose(_to_number(coeff), 0.0, atol=atol)
        }
        return self._new_instance(simplified_data)

    @classmethod
    def _simplify_label(cls, label: str, coeff: _TCoeff) -> tuple[str, _TCoeff]:
        new_label_list = []
        for lbl in label.split()[::-1]:
            index = int(lbl[1:])
            if index not in new_label_list:
                new_label_list.append(index)
            else:
                if (len(new_label_list) - new_label_list.index(index)) % 2 == 0:
                    coeff *= -1
                new_label_list.remove(index)
        new_label_list.reverse()
        return " ".join(map(lambda index: f"_{index}", new_label_list)), coeff
