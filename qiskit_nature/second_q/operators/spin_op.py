# This code is part of a Qiskit project.
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

"""A generic Spin operator.

Note: this implementation differs fundamentally from the ``FermionicOp``
as it relies on the mathematical representation of spin matrices as (e.g.) explained in [1].

[1]: https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from collections import defaultdict
from typing import Iterator, Sequence
from fractions import Fraction
from functools import partial, reduce

import numpy as np

from qiskit_nature import QiskitNatureError

from .polynomial_tensor import PolynomialTensor
from .sparse_label_op import _TCoeff, SparseLabelOp, _to_number


class SpinOp(SparseLabelOp):
    """XYZ Spin operator.

    A ``SpinOp`` represents a weighted sum of spin operator terms with a certain spin
    value associated to them. This value can be an integer for bosonic particles or
    a half-integer (fraction) for fermions.

    These operator terms are encoded as sparse labels, strings consisting of a
    space-separated list of expressions. Each expression must look like
    :code:`[XYZ]_<index>` or :code:`[XYZ]_<index>^<power>`,
    where the :code:`<index>` is a non-negative integer representing the
    index of the spin mode  where the ``X``, ``Y`` or ``Z``
    component of the spin operator is to be applied.

    The value of :code:`index` is bound by the number of spins(``num_spins``) of the operator
    (Note: since Python indices are 0-based, the maximum value an index can take is given by
    :code:`num_spins-1`). The :code:`<power>` is a positive integer indicating the number
    of times the given operator is applied to the mode at :code:`<index>`. You can omit
    :code:`<power>`, implying a single application of the operator (:code:`power = 1`).

    **Initialization**

    A ``SpinOp`` is initialized with a dictionary, mapping terms to their respective
    coefficients. For example:

    .. code-block:: python

        from qiskit_nature.second_q.operators import SpinOp

        x = SpinOp({"X_0": 1}, spin=3/2)
        y = SpinOp({"Y_0": 1}, spin=3/2)
        z = SpinOp({"Z_0": 1}, spin=3/2)

    are :math:`S^x, S^y, S^z` for spin 3/2 system.
    The two qutrit Heisenberg model with transverse magnetic field is

    .. code-block:: python

        SpinOp({
                "X_0 X_1": -1,
                "Y_0 Y_1": -1,
                "Z_0 Z_1": -1,
                "Z_0": -0.3,
                "Z_1": -0.3,
            },
            spin=1
        )

    This means :math:`- S^x_0 S^x_1 - S^y_0 S^y_1 - S^z_0 S^z_1 - 0.3 S^z_0 - 0.3 S^z_1`.

    An example using labels with powers would be:

    .. code-block:: python

        from qiskit_nature.second_q.operators import SpinOp

        op = SpinOp({"X_0^2 Y_1^3 Z_0": 1})


    By default, this way of initializing will create a full copy of the dictionary of coefficients.
    If you have very restricted memory resources available, or would like to avoid the additional
    copy, the dictionary will be stored by reference if you disable ``copy`` like so:

    .. code-block:: python

        some_big_data = {
            "X_0 Y_0": 1.0,
            "X_1 Y_1": -1.0,
            # ...
        }

        op = SpinOp(
            some_big_data,
            num_spins=2,
            copy=False,
        )


    .. note::

        It is the users' responsibility, that in the above scenario, :code:`some_big_data` is not
        changed after initialization of the ``SpinOp``, since the operator contents are not
        guaranteed to remain unaffected by such changes.

    **Algebra**

    :class:`SpinOp` supports the following basic arithmetic operations: addition, subtraction,
    scalar multiplication, adjoint, composition and tensoring.

    As of now, operations that involve two different instances of ``SpinOp`` (i.e. addition,
    subtraction, composition, and tensoring) are only supported for identical spins
    (``op_1.num_spins == op_2.num_spins``).

    For example,

    Addition

    .. code-block:: python

        SpinOp({"X_1": 1}, num_spins=2) + SpinOp({"X_0": 1}, num_spins=2)

    Sum

    .. code-block:: python

        sum(SpinOp({label: 1}, num_spins=3) for label in ["X_0", "Z_1", "X_2 Z_2"])

    Scalar multiplication

    .. code-block:: python

        0.5 * SpinOp({"X_1": 1}, num_spins=2)

    Operator multiplication

    .. code-block:: python

        op1 = SpinOp({"X_0 Z_1": 1}, num_spins=2)
        op2 = SpinOp({"Z_0 X_0 X_1": 1}, num_spins=2)
        print(op1 @ op2)

    Tensor multiplication

    .. code-block:: python

        op = SpinOp({"X_0 Z_1": 1}, num_spins=2)
        print(op ^ op)

    Adjoint

    .. code-block:: python

        SpinOp({"X_0 Z_1": 1j}, num_spins=2).adjoint()


    **Iteration**

    Instances of ``SpinOp`` are iterable. Iterating a SpinOp yields (term, coefficient)
    pairs describing the terms contained in the operator. Labels containing powers/exponents
    will be expanded into multiple (term, coefficient) pairs.

    The following attributes can be set via the initializer but can also be read and updated once
    the ``SpinOp`` object has been constructed.

    Attributes:
        num_spins (int | None): the number of spins on which this operator acts. This is
            considered a lower bound, which means that mathematical operations acting on two or more
            operators will result in a new operator with the maximum number of spins of any
            of the involved operators.
        spin (float | Fraction): positive half-integer (integer or half-odd-integer)
            that represents spin.
    """

    _OPERATION_REGEX = re.compile(r"([XYZ]_\d+(\^\d+)?\s)*[XYZ]_\d+(\^\d+)?")

    def __init__(
        self,
        data: Mapping[str, _TCoeff],
        spin: float | Fraction = Fraction(1, 2),  # TODO: extend to list
        num_spins: int | None = None,
        *,
        copy: bool = True,
        validate: bool = True,
    ):
        r"""
        Args:
            data: label string, list of labels and coefficients. See the label section in
                  the documentation of :class:`SpinOp` for more details.
            spin: positive half-integer (integer or half-odd-integer) that represents spin.
            num_spins: the number spins on which this operator acts.
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
            QiskitNatureError: when spin is not a positive half-integer.
        """
        self.num_spins = num_spins
        spin = Fraction(spin)
        if spin.denominator not in (1, 2) or spin <= 0:
            raise QiskitNatureError(
                f"spin must be a positive half-integer (integer or half-odd-integer), not {spin}."
            )
        self.spin = spin
        super().__init__(data, copy=copy, validate=validate)

    @property
    def register_length(self) -> int:
        if self.num_spins is None:
            max_index = max(int(term[2:]) for key in self._data for term in key.split())
            return max_index + 1

        return self.num_spins

    def _new_instance(self, data: Mapping[str, _TCoeff], *, other: SpinOp | None = None) -> SpinOp:
        num_s = self.num_spins
        spin = self.spin
        if other is not None:
            other_num_s = other.num_spins
            other_spin = other.spin
            if spin != other_spin:
                raise TypeError(
                    f"Invalid operation between operators with different spin"
                    f"values. Found spin_1={spin} and spin_2={other_spin}."
                )
            if num_s is None:
                num_s = other_num_s
            elif other_num_s is not None:
                num_s = max(num_s, other_num_s)

        return self.__class__(data, copy=False, num_spins=num_s)

    def _validate_keys(self, keys: Collection[str]) -> None:
        super()._validate_keys(keys)

        num_s = self.num_spins

        max_index = -1

        for key in keys:
            # 0. explicitly allow the empty key
            if key == "":
                continue

            # 1. validate overall key structure
            if not re.fullmatch(SpinOp._OPERATION_REGEX, key):
                raise QiskitNatureError(f"{key} is not a valid SpinOp label.")

            # 2. validate all indices against register length
            for term in key.split():
                sub_terms = term.split("^")
                # sub_terms[0] is the base, sub_terms[1] is the (optional) exponent
                index = int(sub_terms[0][2:])
                if num_s is None:
                    if index > max_index:
                        max_index = index
                elif index >= num_s:
                    raise QiskitNatureError(
                        f"The index, {index}, from the label, {key}, exceeds the number of "
                        f"spins, {num_s}."
                    )

        self.num_spins = max_index + 1 if num_s is None else num_s

    @classmethod
    def _validate_polynomial_tensor_key(cls, keys: Collection[str]) -> None:
        allowed_chars = {"X", "Y", "Z"}
        for key in keys:
            if set(key) - allowed_chars:
                raise QiskitNatureError(
                    f"The key {key} is invalid. PolynomialTensor keys may only consists of `X`, "
                    "`Y` and `Z` characters, for them to be expandable into a SpinOp."
                )

    @classmethod
    def from_polynomial_tensor(cls, tensor: PolynomialTensor) -> SpinOp:
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

        return cls(data, copy=False, num_spins=tensor.register_length).chop()

    @staticmethod
    def _split_label(label) -> Iterator[tuple[str, int, int]]:
        """Helper method to iterate over label splits.

        Yields:
            A tuple containing the character, index and exponent in each label split.
        """
        for lbl in label.split():
            char, index = lbl.split("_")
            index_split = index.split("^")
            idx = int(index_split[0])
            exp = int(index_split[1]) if len(index_split) > 1 else 1
            yield char, idx, exp

    @classmethod
    def x(cls, spin: float | Fraction = Fraction(1, 2)) -> SpinOp:
        """Constructs the X spin operator for a given spin.

        Returns:
            The X spin operator for ``spin``.
        """
        return cls({"X_0": 1.0}, spin=spin, num_spins=1, copy=False)

    @classmethod
    def y(cls, spin: float | Fraction = Fraction(1, 2)) -> SpinOp:
        """Constructs the Y spin operator for a given spin.

        Returns:
            The Y spin operator for ``spin``.
        """
        return cls({"Y_0": 1.0}, spin=spin, num_spins=1, copy=False)

    @classmethod
    def z(cls, spin: float | Fraction = Fraction(1, 2)) -> SpinOp:
        """Constructs the Z spin operator for a given spin.

        Returns:
            The Z spin operator for ``spin``.
        """
        return cls({"Z_0": 1.0}, spin=spin, num_spins=1, copy=False)

    @classmethod
    def one(cls, spin: float | Fraction = Fraction(1, 2)) -> SpinOp:
        # pylint: disable=arguments-differ
        """Constructs the "one" spin operator for a given spin.

        Returns:
            The "one" spin operator for ``spin``.
        """
        return cls({"": 1.0}, spin=spin, copy=False)

    @classmethod
    def zero(cls, spin: float | Fraction = Fraction(1, 2)) -> SpinOp:
        # pylint: disable=arguments-differ
        """Constructs the "zero" spin operator for a given spin.

        Returns:
            The "zero" spin operator for ``spin``.
        """
        return cls({}, spin=spin, copy=False)

    def __repr__(self) -> str:
        data_str = f"{dict(self.items())}"

        return "SpinOp(" f"{data_str}, " f"spin={self.spin}, " f"num_spins={self.num_spins}, " ")"

    def __str__(self) -> str:
        pre = (
            "Spin Operator\n"
            f"spin={self.spin}, number spins={self.num_spins}, number terms={len(self)}\n"
        )
        ret = "  " + "\n+ ".join(
            [f"{coeff} * ( {label} )" if label else f"{coeff}" for label, coeff in self.items()]
        )
        return pre + ret

    def terms(self) -> Iterator[tuple[list[tuple[str, int]], _TCoeff]]:
        """Provides an iterator analogous to :meth:`items` but with the labels already split into
        pairs of operation characters and indices. If the labels contain an exponent, they will be
        expanded into as many elements as indicated by the exponent. For example, label ``"X_0^3"``
        will yield ``([("X", 0), ("X", 0), ("X", 0)], coeff)``.

        Yields:
            A tuple with two items; the first one being a list of pairs of the form (char, int)
            where char is either ``X``, ``Y`` or ``Z`` and the integer corresponds to the index
            on which the operator gets applied; the second item of the returned tuple is the
            coefficient of this term.
        """
        for label in iter(self):
            if not label:
                yield ([], self[label])
                continue

            terms = []
            for char, index, exp in self._split_label(label):
                terms += [(char, index)] * exp
            yield (terms, self[label])

    def _permute_term(
        self, term: list[tuple[str, int]], permutation: Sequence[int]
    ) -> list[tuple[str, int]]:
        return [(action, permutation[index]) for action, index in term]

    @classmethod
    def from_terms(cls, terms: Sequence[tuple[list[tuple[str, int]], _TCoeff]]) -> SpinOp:
        data = {
            " ".join(f"{action}_{index}" for action, index in label): value
            for label, value in terms
        }
        return cls(data)

    def conjugate(self) -> SpinOp:
        """Returns the conjugate of the ``SpinOp``.

        Returns:
            The complex conjugate of this ``SpinOp``.
        """
        new_data = {}
        for label, coeff in self.items():
            # calculate conjugate of coefficients
            coeff = np.conjugate(coeff)
            for char, _, exp in self._split_label(label):
                # add sign from Y-terms (Y.conj() = -Y)
                if char == "Y" and exp % 2:
                    coeff *= -1
            new_data[label] = coeff

        return self._new_instance(new_data)

    def transpose(self) -> SpinOp:
        """Returns the transpose of the ``SpinOp``.

        Returns:
            The transpose of the ``SpinOp``.
        """
        # note: X^T=X, Y^T=-Y, Z^T=Z, (XY)^T = Y^T X^T
        data = {}
        for label, coeff in self.items():
            for char, _, exp in self._split_label(label):
                # add sign from Y-terms (Y^T=-Y)
                if char == "Y" and exp % 2:
                    coeff *= -1
            data[" ".join(lbl for lbl in reversed(label.split()))] = coeff

        return self._new_instance(data)

    def compose(self, other: SpinOp, qargs=None, front: bool = False) -> SpinOp:
        if not isinstance(other, SpinOp):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SpinOp' and '{type(other).__name__}'"
            )

        if front:
            return self._tensor(self, other, offset=False)
        else:
            return self._tensor(other, self, offset=False)

    def tensor(self, other: SpinOp) -> SpinOp:
        return self._tensor(self, other)

    def expand(self, other: SpinOp) -> SpinOp:
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a: SpinOp, b: SpinOp, *, offset: bool = True) -> SpinOp:
        shift = a.num_spins if offset else 0

        new_data: dict[str, _TCoeff] = {}
        for terms1, cf1 in a.terms():
            label1 = f"{''.join(f' {c}_{i}' for c, i in terms1)}"
            for terms2, cf2 in b.terms():
                new_label = f"{label1} {' '.join(f'{c}_{i + shift}' for c, i in terms2)}".strip()
                if new_label in new_data:
                    new_data[new_label] += cf1 * cf2
                else:
                    new_data[new_label] = cf1 * cf2

        new_op = a._new_instance(new_data, other=b)
        if offset:
            new_op.num_spins = a.num_spins + b.num_spins
        return new_op

    def simplify(self, atol: float | None = None) -> SpinOp:
        atol = self.atol if atol is None else atol

        data = defaultdict(complex)  # type: dict[str, _TCoeff]
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

        new_label = []
        for lbl in label.split():
            # the generator will only yield 1 item
            char, idx, exp = next(self._split_label(lbl))
            new_label += [f"{char}_{idx}"] * exp

        return " ".join(lbl for lbl in new_label if lbl is not None), coeff

    def index_order(self) -> SpinOp:
        """Convert to the equivalent operator with the terms of each label ordered by index.

        Returns a new operator (the original operator is not modified).

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

        new_label = " ".join(f"{term[0]}_{term[1]}" for term in terms)
        return new_label, coeff

    def to_matrix(self) -> np.ndarray:
        # TODO: use scipy.sparse.csr_matrix() and add parameter `sparse: bool`.
        """Convert to dense matrix.

        Returns:
            The matrix (numpy.ndarray with dtype=numpy.complex128)
        """
        dim = int(2 * self.spin + 1)

        x_mat = np.fromfunction(
            lambda i, j: np.where(
                np.abs(i - j) == 1,
                np.sqrt((dim + 1) * (i + j + 1) / 2 - (i + 1) * (j + 1)) / 2,
                0,
            ),
            (dim, dim),
            dtype=np.complex128,
        )
        y_mat = np.fromfunction(
            lambda i, j: np.where(
                np.abs(i - j) == 1,
                1j * (i - j) * np.sqrt((dim + 1) * (i + j + 1) / 2 - (i + 1) * (j + 1)) / 2,
                0,
            ),
            (dim, dim),
            dtype=np.complex128,
        )
        z_mat = np.fromfunction(
            lambda i, j: np.where(i == j, (dim - 2 * i - 1) / 2, 0),
            (dim, dim),
            dtype=np.complex128,
        )
        i_mat = np.eye(dim, dim)

        tensorall = partial(reduce, np.kron)
        char_map = {"X": x_mat, "Y": y_mat, "Z": z_mat}

        # reorder and expand
        simplified_op = self.index_order().simplify()

        # the size of the final matrix needs to adjust for the total number of spins which this
        # operator acts on
        final_dim = dim**self.register_length
        final_matrix = np.zeros((final_dim, final_dim), dtype=np.complex128)

        for label, coeff in simplified_op.items():
            matrix_per_idx = {}
            # after .simplify() all exponents will be 1,
            # so the exp return value from self._split_label()
            # can be safely ignored (dropped into _)
            for char, idx, _ in self._split_label(label):
                # compose all matrices in same index
                if idx not in matrix_per_idx:
                    matrix_per_idx[idx] = i_mat
                matrix_per_idx[idx] = matrix_per_idx[idx] @ char_map.get(char, i_mat)

            # fill out empty indices with identity
            dense_matrix_per_idx = [
                matrix_per_idx.get(i, i_mat) for i in range(self.register_length)
            ]
            # add weighted kronecker product to final matrix
            final_matrix += coeff * tensorall(np.asarray(dense_matrix_per_idx))

        return final_matrix
