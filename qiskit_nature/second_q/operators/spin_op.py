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

"""A generic Spin operator.

Note: this implementation differs fundamentally from the `FermionicOp`
as it relies on the mathematical representation of spin matrices as (e.g.) explained in [1].

[1]: https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, MutableMapping
from collections import defaultdict
from typing import cast, Iterator
from fractions import Fraction
from functools import lru_cache, partial, reduce
from itertools import product
from typing import List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from qiskit.utils.validation import validate_min
from qiskit_nature import QiskitNatureError

# from .second_quantized_op import SecondQuantizedOp
from .polynomial_tensor import PolynomialTensor
from .sparse_label_op import SparseLabelOp

class SpinOp(SparseLabelOp):
    """XYZ-ordered Spin operators.

    # TODO: Update docstring
    **Label**

    Allowed characters for primitives of labels are I, X, Y, Z, (+, and -).

    .. list-table::
        :header-rows: 1

        * - Label
          - Mathematical Representation
          - Meaning
        * - `I`
          - :math:`I`
          - Identity operator
        * - `X`
          - :math:`S^x`
          - :math:`x`-component of the spin operator
        * - `Y`
          - :math:`S^y`
          - :math:`y`-component of the spin operator
        * - `Z`
          - :math:`S^z`
          - :math:`z`-component of the spin operator
        * - `+`
          - :math:`S^+`
          - Raising operator
        * - `-`
          - :math:`S^-`
          - Lowering operator


    A sparse label is a string consisting of a space-separated list of words.
    Each word must look like :code:`[XYZI+-]_<index>^<power>`,
    where the :code:`<index>` is a non-negative integer representing the index of the spin mode
    and the :code:`<power>` is a positive integer indicating the number of times the given operator
    is applied to the mode at :code:`<index>`.
    You can omit :code:`<power>`, implying a single application of the operator (:code:`power = 1`).
    For example,

    .. code-block:: python

        "X_0"
        "Y_0^2"
        "Y_0^2 Z_0^3 X_1^1 Y_1^2 Z_1^2"

    are possible labels.
    For each :code:`index` the operations `X`, `Y` and `Z` can only be specified exclusively in
    this order. `+` and `-` cannot be used with `X` and `Y`
    because ladder operators will be parsed into `X` and `Y`.
    Thus, :code:`"Z_0 X_0"`, :code:`"Z_0 +_0"`, and :code:`"+_0 X_0"` are invalid labels.
    The indices must be ascending order.

    :code:`"+_i -_i"` is supported.
    This pattern is parsed to :code:`+_i -_i = X_i^2 + Y_i^2 + Z_i`.

    **Initialization**

    The :class:`SpinOp` can be initialized by the list of tuples.
    For example,

    .. jupyter-execute::

        from qiskit_nature.second_q.operators import SpinOp

        x = SpinOp("X", spin=3/2)
        y = SpinOp("Y", spin=3/2)
        z = SpinOp("Z", spin=3/2)

    are :math:`S^x, S^y, S^z` for spin 3/2 system.
    Two qutrit Heisenberg model with transverse magnetic field is

    .. jupyter-execute::

        SpinOp(
            [
                ("XX", -1),
                ("YY", -1),
                ("ZZ", -1),
                ("ZI", -0.3),
                ("IZ", -0.3),
            ],
            spin=1
        )

    This means :math:`- S^x_0 S^x_1 - S^y_0 S^y_1 - S^z_0 S^z_1 - 0.3 S^z_0 - 0.3 S^z_1`.

    :class:`SpinOp` can be initialized with internal data structure (`numpy.ndarray`) directly.
    In this case, `data` is a tuple of two elements: `spin_array` and `coeffs`.
    `spin_array` is 3-dimensional `ndarray`. 1st axis has three elements 0, 1, and 2 corresponding
    to x, y, and z.  2nd axis represents the index of terms.
    3rd axis represents the index of register.
    `coeffs` is one-dimensional `ndarray` with the length of the number of terms.

    **Algebra**

    :class:`SpinOp` supports the following basic arithmetic operations: addition, subtraction,
    scalar multiplication, and adjoint.
    For example,

    Raising Operator (addition and scalar multiplication)

    .. jupyter-execute::

        x + 1j * y

    Adjoint

    .. jupyter-execute::

        ~(1j * z)

    """

    _OPERATION_REGEX = re.compile(r"([XYZI]_\d+\s)*[XYZI]_\d+")

    def __init__(
        self,
        data: Mapping[str, complex],
        spin: float | Fraction = Fraction(1, 2), # TODO: extend to list
        num_orbitals: int | None = None,
        *,
        copy: bool = True,
        validate: bool = True,
    ):
        r"""
        Args:
            data: label string, list of labels and coefficients. See the label section in
                  the documentation of :class:`SpinOp` for more details.
            spin: positive half-integer (integer or half-odd-integer) that represents spin.
            num_orbitals: the number orbitals on which this operator acts.
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
            QiskitNatureError: when spin is not a positive half-integer.
        """
        self.num_orbitals = num_orbitals
        spin = Fraction(spin)
        if spin.denominator not in (1, 2):
            raise QiskitNatureError(
                f"spin must be a positive half-integer (integer or half-odd-integer), not {spin}."
            )
        self.spin = spin
        super().__init__(data, copy=copy, validate=validate)

    @property
    def register_length(self) -> int | None:
        return self.num_orbitals

    def _new_instance(
        self, data: Mapping[str, complex], *, other: SpinOp | None = None
    ) -> SpinOp:
        num_o = self.num_orbitals
        spin = self.spin
        if other is not None:
            other_num_o = other.num_orbitals
            other_spin = other.spin
            if spin != other_spin:
                raise TypeError(f"Invalid operation between operators with different spin"
                                f"values. Found spin_1={spin} and spin_2={other_spin}.")
            if num_o is None:
                num_o = other_num_o
            elif other_num_o is not None:
                num_o = max(num_o, other_num_o)

        return self.__class__(data, copy=False, num_orbitals=num_o)

    def _validate_keys(self, keys: Collection[str]) -> None:
        super()._validate_keys(keys)

        num_o = self.num_orbitals

        max_index = -1

        for key in keys:
            # 0. explicitly allow the empty key
            if key == "":
                continue

            # 1. validate overall key structure
            if not re.fullmatch(SpinOp._OPERATION_REGEX, key):
                raise QiskitNatureError(f"{key} is not a valid SpinOp label.")

            # 2. validate all indices against register length
            for term in key.split(" "):
                index = int(term[2:])
                if num_o is None:
                    if index > max_index:
                        max_index = index
                elif index >= num_o:
                    raise QiskitNatureError(
                        f"The index, {index}, from the label, {key}, exceeds the number of "
                        f"orbitals, {num_o}."
                    )
            # TODO: do we need to validate the exponent?

        self.num_orbitals = max_index + 1 if num_o is None else num_o

    @classmethod
    def _validate_polynomial_tensor_key(cls, keys: Collection[str]) -> None:
        allowed_chars = {"I", "X", "Y", "Z"}
        for key in keys:
            if set(key) - allowed_chars:
                raise QiskitNatureError(
                    f"The key {key} is invalid. PolynomialTensor keys may only consists of `I`, `X`, "
                    "`Y` and `Z` characters, for them to be expandable into a SpinOp."
                )

    @classmethod
    def from_polynomial_tensor(cls, tensor: PolynomialTensor) -> SpinOp:
        cls._validate_polynomial_tensor_key(tensor.keys())

        data: dict[str, complex] = {}

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

        return "SpinOp(" f"{data_str}, " f"spin={self.spin}, "f"num_orbitals={self.num_orbitals}, " ")"

    def __str__(self) -> str:
        pre = (
            "Spin Operator\n"
            f"spin={self.spin}, number orbitals={self.num_orbitals}, number terms={len(self)}\n"
        )
        ret = "  " + "\n+ ".join(
            [f"{coeff} * ( {label} )" if label else f"{coeff}" for label, coeff in self.items()]
        )
        return pre + ret

    def terms(self) -> Iterator[tuple[list[tuple[str, int]], complex]]:
        """Provides an iterator analogous to :meth:`items` but with the labels already split into
        pairs of operation characters and indices.

        Yields:
            # TODO: update docstring to spin
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
            #   lbl[0] is X, Y, Z or I
            #   lbl[2:] corresponds to the index
            terms = [(lbl[0], int(lbl[2:])) for lbl in label.split(" ")]
            yield (terms, self[label])

    def conjugate(self) -> SpinOp:
        """Returns the conjugate of the ``SpinOp``.

        Returns:
            The complex conjugate of the starting ``SpinOp``.
        """
        new_data = {}
        for label, coeff in self.items():
            num_ys = label.count("Y")
            # calculate conjugate of coefficients
            coeff = np.conjugate(coeff)
            # add sign from Y-terms (Y.conj() = -Y)
            coeff *= -1 if num_ys % 2 != 0 else 1
            new_data[label] = coeff

        return self._new_instance(new_data)

    def transpose(self) -> SpinOp:
        """Returns the transpose of the ``SpinOp``.

        Returns:
            The transpose of the ``SpinOp``.
        """
        # note: X^T=X, Y^T=-Y, Z^T=Z
        # note: (XY)^T = Y^T X^T
        data = {}
        for label, coeff in self.items():
            num_ys = label.count("Y")
            # add sign from Y-terms (Y^T=-Y)
            coeff *= -1 if num_ys % 2 != 0 else 1
            data[" ".join(lbl for lbl in reversed(label.split(" ")))] = coeff

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
        shift = a.num_orbitals if offset else 0

        new_data: dict[str, complex] = {}
        for label1, cf1 in a.items():
            for terms2, cf2 in b.terms():
                new_label = f"{label1} {' '.join(f'{c}_{i + shift}' for c, i in terms2)}".strip()
                if new_label in new_data:
                    new_data[new_label] += cf1 * cf2
                else:
                    new_data[new_label] = cf1 * cf2

        new_op = a._new_instance(new_data, other=b)
        if offset:
            new_op.num_orbitals = a.num_orbitals + b.num_orbitals
        return new_op

    def simplify(self, *, atol: float | None = None, reorder: bool = False) -> SpinOp:
        atol = self.atol if atol is None else atol

        data = defaultdict(complex)  # type: dict[str, complex]
        for label, coeff in self.items():
            label, coeff = self._simplify_label(label, coeff, reorder)
            data[label] += coeff
        simplified_data = {
            label: coeff for label, coeff in data.items() if not np.isclose(coeff, 0.0, atol=atol)
        }
        return self._new_instance(simplified_data)

    def _simplify_label(self, label: str, coeff: complex, reorder: bool = False) -> tuple[str, complex]:

        bits = {}
        new_label = []
        for i, lbl in enumerate(label.split()):
            char = lbl[0]
            idx = int(lbl[2:])

            if idx in bits and bits[idx][-1] == char:
                bits[idx][-1] = ""
                # remove last instance of lbl
                new_label.reverse()
                new_label.remove(lbl)
                new_label.reverse()
            else:
                if idx not in bits:
                    bits[idx] = [char]
                else:
                    bits[idx].append(char)
                new_label.append(lbl)

        if reorder:
            new_label = []
            for idx in sorted(bits):
                new_label.extend([f"{char}_{idx}" for char in bits[idx] if char != ""])

        return " ".join(lbl for lbl in new_label if lbl is not None), coeff

