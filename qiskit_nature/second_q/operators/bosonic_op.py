# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Bosonic-particle Operator."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Iterator, Sequence

import numpy as np

from qiskit_nature.exceptions import QiskitNatureError

from .polynomial_tensor import PolynomialTensor
from .sparse_label_op import _TCoeff, SparseLabelOp, _to_number


class BosonicOp(SparseLabelOp):
    r"""N-mode Bosonic operator.

    A ``BosonicOp`` represents a weighted sum of bosonic creation/annihilation operator terms.
    These terms are encoded as sparse labels, which are strings consisting of a space-separated list of
    expressions. Each expression must look like :code:`[+-]_<index>`, where the :code:`<index>` is a
    non-negative integer representing the index of the bosonic mode where the ``+`` (creation) or
    ``-`` (annihilation) operation is to be performed. The value of :code:`index` is bound by the
    number of modes (``num_modes``) of the operator (Note: since Python indices are
    0-based, the maximum value an index can take is given by :code:`num_modes-1`).

    **Initialization**

    A ``BosonicOp`` is initialized with a dictionary, mapping terms to their respective
    coefficients:

    .. code-block:: python

        from qiskit_nature.second_q.operators import BosonicOp

        op = BosonicOp(
            {
                "+_0 -_0": 1.0,
                "+_1 -_1": -1.0,
            },
            num_modes=2,
        )

    By default, this way of initializing will create a full copy of the dictionary of coefficients.
    If you have very restricted memory resources available, or would like to avoid the additional
    copy, the dictionary will be stored by reference if you disable ``copy`` like so:

    .. code-block:: python

        some_big_data = {
            "+_0 -_0": 1.0,
            "+_1 -_1": -1.0,
            # ...
        }

        op = BosonicOp(
            some_big_data,
            num_modes=2,
            copy=False,
        )


    .. note::

        It is the users' responsibility, that in the above scenario, :code:`some_big_data` is not
        changed after initialization of the `BosonicOp`, since the operator contents are not
        guaranteed to remain unaffected by such changes.

    **Algebra**

    This class supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, operator multiplication, and adjoint.
    For example,

    Addition

    .. code-block:: python

      BosonicOp({"+_1": 1}, num_modes=2) + BosonicOp({"+_0": 1}, num_modes=2)

    Sum

    .. code-block:: python

      sum(BosonicOp({label: 1}, num_modes=3) for label in ["+_0", "-_1", "+_2 -_2"])

    Scalar multiplication

    .. code-block:: python

      0.5 * BosonicOp({"+_1": 1}, num_modes=2)

    Operator multiplication

    .. code-block:: python

      op1 = BosonicOp({"+_0 -_1": 1}, num_modes=2)
      op2 = BosonicOp({"-_0 +_0 +_1": 1}, num_modes=2)
      print(op1 @ op2)

    Tensor multiplication

    .. code-block:: python

      op = BosonicOp({"+_0 -_1": 1}, num_modes=2)
      print(op ^ op)

    Adjoint

    .. code-block:: python

      BosonicOp({"+_0 -_1": 1j}, num_modes=2).adjoint()

    **Iteration**

    Instances of ``BosonicOp`` are iterable. Iterating a ``BosonicOp`` yields (term, coefficient)
    pairs describing the terms contained in the operator.

    Attributes:
        num_modes (int | None): the number of modes on which this operator acts. This is considered
            a lower bound, which means that mathematical operations acting on two or more operators
            will result in a new operator with the maximum number of modes of any of the involved
            operators.

    .. note::

        A BosonicOp can contain :class:`qiskit.circuit.ParameterExpression` objects as coefficients.
        However, a BosonicOp containing parameters does not support the following methods:

        - ``is_hermitian``
    """

    _OPERATION_REGEX = re.compile(r"([\+\-]_\d+\s)*[\+\-]_\d+")

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
        self.num_modes = num_modes
        super().__init__(data, copy=copy, validate=validate)

    @property
    def register_length(self) -> int:
        if self.num_modes is None:
            max_index = max(int(term[2:]) for key in self._data for term in key.split())
            return max_index + 1

        return self.num_modes

    def _new_instance(
        self, data: Mapping[str, _TCoeff], *, other: BosonicOp | None = None
    ) -> BosonicOp:
        num_so = self.num_modes
        if other is not None:
            other_num_so = other.num_modes
            if num_so is None:
                num_so = other_num_so
            elif other_num_so is not None:
                num_so = max(num_so, other_num_so)

        return self.__class__(data, copy=False, num_modes=num_so)

    def _validate_keys(self, keys: Collection[str]) -> None:
        super()._validate_keys(keys)

        num_so = self.num_modes

        max_index = -1

        for key in keys:
            # 0. explicitly allow the empty key
            if key == "":
                continue

            # 1. validate overall key structure
            if not re.fullmatch(BosonicOp._OPERATION_REGEX, key):
                raise QiskitNatureError(f"{key} is not a valid BosonicOp label.")

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

        self.num_modes = max_index + 1 if num_so is None else num_so

    @classmethod
    def _validate_polynomial_tensor_key(cls, keys: Collection[str]) -> None:
        allowed_chars = {"+", "-"}

        for key in keys:
            if set(key) - allowed_chars:
                raise QiskitNatureError(
                    f"The key {key} is invalid. PolynomialTensor keys may only consists of `+` and "
                    "`-` characters, for them to be expandable into a BosonicOp."
                )

    @classmethod
    def from_polynomial_tensor(cls, tensor: PolynomialTensor) -> BosonicOp:
        cls._validate_polynomial_tensor_key(tensor.keys())

        data: dict[str, _TCoeff] = {}

        for key in tensor:
            if key == "":
                data[""] = tensor[key].item()
                continue

            mat = tensor[key]

            label_template = mat.label_template.format(*key)

            for value, index in mat.coord_iter():
                data[label_template.format(*index)] = value

        return cls(data, copy=False, num_modes=tensor.register_length).chop()

    def __repr__(self) -> str:
        data_str = f"{dict(self.items())}"

        return "BosonicOp(" f"{data_str}, " f"num_modes={self.num_modes}, " ")"

    def __str__(self) -> str:
        pre = "Bosonic Operator\n" f"number modes={self.num_modes}, number terms={len(self)}\n"
        ret = "  " + "\n+ ".join(
            [f"{coeff} * ( {label} )" if label else f"{coeff}" for label, coeff in self.items()]
        )
        return pre + ret

    def terms(self) -> Iterator[tuple[list[tuple[str, int]], _TCoeff]]:
        """Provides an iterator analogous to :meth:`items` but with the labels already split into
        pairs of operation characters and indices.

        Yields:
            A tuple with two items; the first one being a list of pairs of the form (char, int)
            where char is either `+` or `-` and the integer corresponds to the bosonic mode index
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

    @classmethod
    def from_terms(cls, terms: Sequence[tuple[list[tuple[str, int]], _TCoeff]]) -> BosonicOp:
        data = {
            " ".join(f"{action}_{index}" for action, index in label): value
            for label, value in terms
        }
        return cls(data)

    def _permute_term(
        self, term: list[tuple[str, int]], permutation: Sequence[int]
    ) -> list[tuple[str, int]]:
        return [(action, permutation[index]) for action, index in term]

    def compose(self, other: BosonicOp, qargs=None, front: bool = False) -> BosonicOp:
        if not isinstance(other, BosonicOp):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'BosonicOp' and '{type(other).__name__}'"
            )

        if front:
            return self._tensor(self, other, offset=False)
        else:
            return self._tensor(other, self, offset=False)

    def tensor(self, other: BosonicOp) -> BosonicOp:
        return self._tensor(self, other)

    def expand(self, other: BosonicOp) -> BosonicOp:
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a: BosonicOp, b: BosonicOp, *, offset: bool = True) -> BosonicOp:
        shift = a.num_modes if offset else 0

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
            new_op.num_modes = a.num_modes + b.num_modes
        return new_op

    def transpose(self) -> BosonicOp:
        data = {}

        trans = "".maketrans("+-", "-+")

        for label, coeff in self.items():
            data[" ".join(lbl.translate(trans) for lbl in reversed(label.split()))] = coeff

        return self._new_instance(data)

    def normal_order(self) -> BosonicOp:
        """Convert to the equivalent operator in normal order.

        The normal order for bosons is defined
        [here](https://en.wikipedia.org/wiki/Normal_order#Bosons).

        Returns:
            A new normal ordered BosonicOp (the original operator is not modified).

        .. note:

            The commutation relation between two bosonic operator is:
        .. math:
            [b_i, b_j^\\dagger]_- = \\delta_{ij} = \\
            b_i * b_j^\\dagger = b_j^\\dagger * b_i + \\delta_{ij}

        .. note::

            This method implements the transformation of an operator to the normal ordered operator.
            The transformation is calculated by considering all commutation relations between the
            operators.
            For example, for the case :math:`\\colon b_0 b_0^\\dagger\\colon` where :math:`b_0`
            is an annihilation operator, this method returns :math:`1 + b_0^\\dagger b_0` due to
            commutation relations.
            See the reference: [here](https://en.wikipedia.org/wiki/Normal_order#Multiple_bosons).
        """
        ordered_op = BosonicOp.zero()

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

    def _normal_order(self, terms: list[tuple[str, int]], coeff: _TCoeff) -> BosonicOp:
        if not terms:
            return self._new_instance({"": coeff})

        ordered_op = BosonicOp.zero()

        # perform insertion sorting
        for i in range(1, len(terms)):
            for j in range(i, 0, -1):
                right = terms[j]
                left = terms[j - 1]

                if right[0] == "+" and left[0] == "-":
                    # swap terms where an annihilation operator is left of a creation operator
                    terms[j - 1] = right
                    terms[j] = left
                    # No need to multiply the coefficient by -1 due to commutation relations

                    if right[1] == left[1]:
                        # if their indices are identical, we incur an additional term because of:
                        # b_i b_i^\dagger = 1 + b_i^\dagger b_i
                        new_terms = terms[: (j - 1)] + terms[(j + 1) :]
                        # we can do so by recursion on this method
                        ordered_op += self._normal_order(new_terms, coeff)

                elif right[0] == left[0] and left[1] > right[1]:
                    # when we have identical neighboring operators.
                    # If the left index is higher than the right one, swap the terms. Else nothing to do
                    terms[j - 1] = right
                    terms[j] = left
                    # No need to multiply the coefficient by -1 due to commutation relations

        new_label = " ".join(f"{term[0]}_{term[1]}" for term in terms)
        ordered_op += self._new_instance({new_label: coeff})
        return ordered_op

    def index_order(self) -> BosonicOp:
        """Convert to the equivalent operator with the terms of each label ordered by index.

        Returns:
            A new index ordered BosonicOp (the original operator is not modified).

        .. note::

            You can use this method to achieve the most aggressive simplification of an operator
            without changing the operation order per index. :meth:`simplify` does *not* reorder the
            terms and, thus, cannot deduce ``-_0 +_1`` and ``+_1 -_0`` to be identical labels.
            Calling this method will reorder the latter label to ``-_0 +_1``, correctly collapsing
            these two labels into one.
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
                    # No need to multiply the coefficient by -1 due to commutation relations

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

    def simplify(self, atol: float | None = None) -> BosonicOp:
        """Simplify the operator.

        The simplifications implemented by this method should be:
        - to eliminate terms whose coefficients are close (w.r.t. ``atol``) to 0.
        - to combine the coefficients which correspond to equivalent terms (see also the note below)

        .. note::

            :meth:`simplify` should be used to simplify terms whose coefficients are close to zero,
            up to the specified numerical tolerance. It still differs slightly from :meth:`chop`
            because that will chop real and imaginary part components individually.

        .. note::

           The meaning of "equivalence" between multiple terms depends on the specific operator
           subclass. As a restriction this method is required to preserve the order of appearance of
           the different components within a term. This avoids some possibly unexpected edge cases.
           However, this also means that some equivalencies cannot be detected. Check for other
           methods of a specific subclass which may affect the order of terms and can allow for
           further simplifications to be implemented. For example, check out :meth:`index_order`.

        .. note::

            :meth:`simplify` is not allowed to simplify the labels ``+_0 -_0`` or ``-_0 +_0``. The
            former corresponds to the boson number operator, and when it is applied to a state yields
            the number of bosons in that state :math:`n`. Similarly, the latter yields
            :math:`1 + n`. As a consequence, the label ``+_0 -_0 -_1 +_0`` will remain untouched by
            this method. This is in contrast to how :meth:`.FermionicOp.simplify` works,
            because it exploits that :math:`n` can be either :math:`0` or :math:`1`.

        This method returns a new operator (the original operator is not modified).

        Args:
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``.

        Returns:
            The simplified operator.
        """
        atol = self.atol if atol is None else atol

        simplified_data = {
            label: coeff
            for label, coeff in self.items()
            if not np.isclose(_to_number(coeff), 0.0, atol=atol)
        }
        return self._new_instance(simplified_data)
