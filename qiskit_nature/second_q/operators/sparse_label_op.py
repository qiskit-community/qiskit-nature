# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Sparse Label Operator base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from numbers import Number
from typing import Iterator, Sequence

import cmath
import numpy as np

from qiskit.quantum_info.operators.mixins import (
    AdjointMixin,
    GroupMixin,
    LinearMixin,
    TolerancesMixin,
)


class SparseLabelOp(LinearMixin, AdjointMixin, GroupMixin, TolerancesMixin, ABC, Mapping):
    """The Sparse Label Operator base class."""

    def __init__(
        self,
        data: Mapping[str, complex],
        register_length: int,
        *,
        copy: bool = True,
        validate: bool = True,
    ):
        """
        Args:
            data: the operator data, mapping string-based keys to numerical values.
            register_length: the length of the operators register. This coincides with the maximum
                index on which an operation may be performed by this operator.
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
        self._data: Mapping[str, complex] = {}
        if copy:
            if validate:
                self.__class__._validate_keys(data.keys(), register_length)
            self._data = dict(data.items())
        else:
            self._data = data
        self._register_length = register_length

    @property
    def register_length(self) -> int:
        """Returns the register length"""
        return self._register_length

    @classmethod
    @abstractmethod
    def _validate_keys(cls, keys: Collection[str], register_length: int) -> None:
        """Validates a key.

        Args:
            keys: the keys to validate.
            register_length: the register length of the operator.

        Raises:
            QiskitNatureError: when an invalid key is encountered.
        """

    @classmethod
    def zero(cls, register_length: int) -> SparseLabelOp:
        """Constructs a zero-operator.

        Args:
            register_length: the length of the operator.

        Returns:
            The zero-operator of the given length.
        """
        return cls({}, register_length=register_length, copy=False)

    @classmethod
    def one(cls, register_length: int) -> SparseLabelOp:
        """Constructs a unity-operator.

        Args:
            register_length: the length of the operator.

        Returns:
            The unity-operator of the given length.
        """
        return cls({"": 1.0}, register_length=register_length, copy=False)

    # workaround until Qiskit Terra 0.22 is released
    def __radd__(self, other):
        return self.__add__(other)

    def _add(self, other: SparseLabelOp, qargs: None = None) -> SparseLabelOp:
        """Return Operator addition of self and other.

        Args:
            other: the second ``SparseLabelOp`` to add to the first.
            qargs: UNUSED.

        Returns:
            The new summed ``SparseLabelOp``.

        Raises:
            ValueError: when ``qargs`` argument is not ``None``
        """
        # workaround until Qiskit Terra 0.22 is released
        if other == 0:
            return self

        if not isinstance(other, SparseLabelOp):
            raise ValueError(
                f"Unsupported operand type(s) for +: 'SparseLabelOp' and '{type(other).__name__}'"
            )

        new_data = {key: value + other._data.get(key, 0) for key, value in self._data.items()}
        other_unique = {key: other._data[key] for key in other._data.keys() - self._data.keys()}
        new_data.update(other_unique)

        register_length = max(self.register_length, other.register_length)

        return self.__class__(new_data, register_length, copy=False)

    def _multiply(self, other: complex) -> SparseLabelOp:
        """Return scalar multiplication of self and other.

        Args:
            other: the number to multiply the ``SparseLabelOp`` values by.

        Returns:
            The newly multiplied ``SparseLabelOp``.

        Raises:
            TypeError: if ``other`` is not compatible type (int, float or complex)
        """
        if not isinstance(other, Number):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SparseLabelOp' and '{type(other).__name__}'"
            )
        new_data = {key: val * other for key, val in self._data.items()}

        return self.__class__(new_data, self.register_length, copy=False)

    def conjugate(self) -> SparseLabelOp:
        """Returns the conjugate of the ``SparseLabelOp``.

        Returns:
            The complex conjugate of the starting ``SparseLabelOp``.
        """
        new_data = {key: np.conjugate(val) for key, val in self._data.items()}

        return self.__class__(new_data, self.register_length, copy=False)

    @abstractmethod
    def transpose(self) -> SparseLabelOp:
        """Returns the transpose of the ``SparseLabelOp``.

        Returns:
            The transpose of the starting ``SparseLabelOp``.
        """

    @abstractmethod
    def compose(
        self, other: SparseLabelOp, qargs: None = None, front: bool = False
    ) -> SparseLabelOp:
        r"""Returns the operator composition with another SparseLabelOp.

        Args:
            other: the other SparseLabelOp.
            qargs: UNUSED.
            front: If True composition uses right operator multiplication, otherwise left
                multiplication is used (the default).

        Returns:
            The operator resulting from the composition.

        .. note::
            Composition (``&``) by default is defined as `left` matrix multiplication for
            matrix operators, while ``@`` (equivalent to :meth:`dot`) is defined as `right` matrix
            multiplication. This means that ``A & B == A.compose(B)`` is equivalent to
            ``B @ A == B.dot(A)`` when ``A`` and ``B`` are of the same type.

            Setting the ``front=True`` keyword argument changes this to `right` matrix
            multiplication which is equivalent to the :meth:`dot` method
            ``A.dot(B) == A.compose(B, front=True)``.
        """

    @abstractmethod
    def tensor(self, other: SparseLabelOp) -> SparseLabelOp:
        r"""Returns the tensor product with another SparseLabelOp.

        Args:
            other: the other SparseLabelOp.

        Returns:
            The operator resulting from the tensor product, :math:`self \otimes other`.

        .. note::
            The tensor product can be obtained using the ``^`` binary operator.
            Hence ``a.tensor(b)`` is equivalent to ``a ^ b``.

        .. note:
            Tensor uses reversed operator ordering to :meth:`expand`.
            For two operators of the same type ``a.tensor(b) = b.expand(a)``.
        """
        return self

    @abstractmethod
    def expand(self, other: SparseLabelOp) -> SparseLabelOp:
        r"""Returns the reverse-order tensor product with another SparseLabelOp.

        Args:
            other: the other SparseLabelOp.

        Returns:
            The operator resulting from the tensor product, :math:`othr \otimes self`.

        .. note:
            Expand is the opposite operator ordering to :meth:`tensor`.
            For two operators of the same type ``a.expand(b) = b.tensor(a)``.
        """
        return self

    def equiv(
        self, other: SparseLabelOp, *, atol: float | None = None, rtol: float | None = None
    ) -> bool:
        """Check equivalence of two ``SparseLabelOp`` instances up to an accepted tolerance.

        The absolute and relative tolerances can be changed via the `atol` and `rtol` attributes,
        respectively.

        Args:
            other: the second ``SparseLabelOp`` to compare with this instance.
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``.
            rtol: Relative numerical tolerance. The default behavior is to use ``self.rtol``.

        Returns:
            True if operators are equivalent, False if not.
        """
        if not isinstance(other, SparseLabelOp):
            return False
        if self.register_length != other.register_length:
            return False
        if self._data.keys() != other._data.keys():
            return False

        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else rtol
        for key, value in self._data.items():
            if not cmath.isclose(value, other._data[key], rel_tol=rtol, abs_tol=atol):
                return False

        return True

    def __eq__(self, other: object) -> bool:
        """Check exact equality of two ``SparseLabelOp`` instances

        Args:
            other: the second ``SparseLabelOp`` to compare with this instance.

        Returns:
            True if operators are equal, False if not.
        """
        if not isinstance(other, SparseLabelOp):
            return False
        return self.register_length == other.register_length and self._data == other._data

    def __getitem__(self, __k: str) -> complex:
        """Get the requested element of the ``SparseLabelOp``."""
        return self._data.__getitem__(__k)

    def __len__(self) -> int:
        """Returns the length of the ``SparseLabelOp``."""
        return self._data.__len__()

    def __iter__(self) -> Iterator[str]:
        """An iterator over the keys of the ``SparseLabelOp``."""
        return self._data.__iter__()

    @abstractmethod
    def terms(self) -> Iterator[tuple[list[tuple[str, int]], complex]]:
        """Provides an iterator analogous to :meth:`items` but with the labels already split into
        pairs of operation characters and indices.

        Yields:
            A tuple with two items; the first one being a list of pairs of the form (char, int)
            where char represents the operation applied to the register and the integer corresponds
            to the register index on which the operator gets applied; the second item of the
            returned tuple is the coefficient of this term.
        """

    def __pow__(self, power):
        if power == 0:
            return self.__class__.one(self.register_length)

        return super().__pow__(power)

    def argsort(self, *, weight: bool = False) -> Sequence[str]:
        """Returns the keys which sort this operator.

        Args:
            weight: when True, the returned keys will sort this operator according to the
                coefficient weights of the stored terms; when False, the keys will sort the operator
                by its keys (i.e. lexicographically).

        Returns:
            The sequence of keys which sort this operator.
        """
        key = self.get if weight else None
        return sorted(self, key=key)

    def sort(self, *, weight: bool = False) -> SparseLabelOp:
        """Returns a new sorted operator.

        Args:
            weight: when True, the returned keys will sort this operator according to the
                coefficient weights of the stored terms; when False, the keys will sort the operator
                by its keys (i.e. lexicographically).

        Returns:
            A new operator instance with its contents sorted.
        """
        indices = self.argsort(weight=weight)
        return self.__class__(
            {ind: self[ind] for ind in indices}, register_length=self.register_length, copy=False
        )

    def induced_norm(self, order: int = 1) -> float:
        r"""Returns the p-norm induced by the operator coefficients.

        If the operator is represented as a sum of terms

        .. math::
            \sum_i w_i H_i

        then the induced :math:`p`-norm is

        .. math::
            \left(\sum_i |w_i|^p \right)^{1/p}

        This is the standard :math:`p`-norm of the operator coefficients
        considered as a vector (see `https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm`_).
        Note that this method does not normal-order or simplify the operator
        before computing the norm; performing either of those operations
        can affect the result.

        Args:
            order: Order :math:`p` of the norm. The default value is 1.

        Returns:
            The induced norm.

        .. _https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm:
            https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm
        """
        return sum(abs(coeff) ** order for coeff in self.values()) ** (1 / order)
