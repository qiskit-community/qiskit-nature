# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
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
from numbers import Complex
from typing import Iterator, Sequence, Union

import cmath
import numpy as np
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info.operators.mixins import (
    AdjointMixin,
    GroupMixin,
    LinearMixin,
    TolerancesMixin,
)

from .polynomial_tensor import PolynomialTensor


_TCoeff = Union[complex, ParameterExpression]  # pylint: disable=invalid-name


def _to_number(a: _TCoeff) -> complex:
    if isinstance(a, ParameterExpression):
        sympified = a.sympify()
        return complex(sympified) if sympified.is_Number else np.nan
    return a


class SparseLabelOp(LinearMixin, AdjointMixin, GroupMixin, TolerancesMixin, ABC, Mapping):
    """The base class for sparse second-quantized operators.

    This class generalizes the storing of operators down to a mapping from string-based labels to
    coefficients. No assumptions are to be made about the contents of the string keys, which is
    entirely left up to concrete subclasses of this base.

    Since this is an abstract base class, we cannot show concrete computation examples here, but
    operators implementing this interface will support:

    - linear operations such as addition and scalar multiplication
    - operator multiplication such as composition and tensor products
    - transposition, conjugation and (by extension) the :meth:`adjoint`
    - equality and equivalence (using the :attr:`atol` and :attr:`rtol` tolerances) comparisons

    Furthermore, several general utility methods exist which are documented below.

    .. note::

        A SparseLabelOp can contain :class:`qiskit.circuit.ParameterExpression` objects as coefficients.
        However, a SparseLabelOp containing parameters does not support the following methods:

        - ``equiv``
        - ``induced_norm``
    """

    # ensure operational compatibility with numpy numeric types
    __array_priority__ = 20

    def __init__(
        self,
        data: Mapping[str, _TCoeff],
        *,
        copy: bool = True,
        validate: bool = True,
    ) -> None:
        """
        Args:
            data: the operator data, mapping string-based keys to numerical values.
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
        self._data: Mapping[str, _TCoeff] = {}
        if copy:
            if validate:
                self._validate_keys(data.keys())
            self._data = dict(data.items())
        else:
            self._data = data

    @property
    @abstractmethod
    def register_length(self) -> int:
        """Returns the register length"""

    @abstractmethod
    def _new_instance(
        self,
        data: Mapping[str, _TCoeff],
        *,
        other: SparseLabelOp | None = None,
    ) -> SparseLabelOp:
        """A utility method constructing a new operator instance from self and the provided data.

        This method should be used whenever the class constructor would need to be called. It
        handles the operator instance construction in a single place with an unchanged signature
        compared to the potentially varying number of arguments on concrete implementations of this
        abstract base operator class.

        To be more concrete, this method should construct a new operator with the provided ``data``
        using otherwise unchanged additional information from ``self``.
        This method can be overwritten by a subclass in order to allow handling of additional data
        from ``self`` and ``other``, an optional second operator instance.
        To provide a concrete example, lower bounded values such as those which yield the
        :attr:`register_length` attribute can be dealt with in a consistent way.

        Args:
            data: the new data from which to construct the new operator instance.
            other: an optional second operator instance.

        Returns:
            The new operator instance.
        """

    @abstractmethod
    def _validate_keys(self, keys: Collection[str]) -> None:
        """Validates the keys of the operator.

        Args:
            keys: the keys to validate.

        Raises:
            QiskitNatureError: when an invalid key is encountered.
        """

    @classmethod
    @abstractmethod
    def from_polynomial_tensor(cls, tensor: PolynomialTensor) -> SparseLabelOp:
        """Constructs the operator from a :class:`~.PolynomialTensor`.

        Args:
            tensor: the :class:`~.PolynomialTensor` to be expanded.

        Returns:
            The constructed operator.
        """

    @classmethod
    @abstractmethod
    def _validate_polynomial_tensor_key(cls, keys: Collection[str]) -> None:
        """Validates the keys of a :class:`~.PolynomialTensor` to be expanded into a ``SparseLabelOp``.

        Args:
            keys: the keys to validate.

        Raises:
            QiskitNatureError: when an invalid key is encountered.
        """

    @classmethod
    def zero(cls) -> SparseLabelOp:
        """Constructs a zero-operator.

        Returns:
            The zero-operator of the given length.
        """
        return cls({}, copy=False)

    @classmethod
    def one(cls) -> SparseLabelOp:
        """Constructs a unity-operator.

        Returns:
            The unity-operator of the given length.
        """
        return cls({"": 1.0}, copy=False)

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
        if not isinstance(other, self.__class__):
            raise ValueError(
                f"Unsupported operand type(s) for +: '{type(self)}' and '{type(other).__name__}'"
            )

        new_data = {key: value + other._data.get(key, 0) for key, value in self._data.items()}
        other_unique = {key: other._data[key] for key in other._data.keys() - self._data.keys()}
        new_data.update(other_unique)

        return self._new_instance(new_data, other=other)

    def _multiply(self, other: _TCoeff) -> SparseLabelOp:
        """Return scalar multiplication of self and other.

        Args:
            other: the number to multiply the ``SparseLabelOp`` values by.

        Returns:
            The newly multiplied ``SparseLabelOp``.

        Raises:
            TypeError: if ``other`` is not compatible type (int, float or complex)
        """
        if not isinstance(other, (Complex, ParameterExpression)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'SparseLabelOp' and '{type(other).__name__}'"
            )
        new_data = {key: val * other for key, val in self._data.items()}

        return self._new_instance(new_data)

    def conjugate(self) -> SparseLabelOp:
        """Returns the conjugate of the ``SparseLabelOp``.

        Returns:
            The complex conjugate of the starting ``SparseLabelOp``.
        """
        new_data = {key: np.conjugate(val) for key, val in self._data.items()}

        return self._new_instance(new_data)

    @abstractmethod
    def transpose(self) -> SparseLabelOp:
        """Returns the transpose of the operator.

        Returns:
            The transpose of the operator.
        """

    @abstractmethod
    def compose(
        self, other: SparseLabelOp, qargs: None = None, front: bool = False
    ) -> SparseLabelOp:
        r"""Returns the operator composition with another operator.

        Args:
            other: the other operator.
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
        r"""Returns the reverse-order tensor product with another operator.

        Args:
            other: the other operator.

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

        Args:
            other: the second ``SparseLabelOp`` to compare with this instance.
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``.
            rtol: Relative numerical tolerance. The default behavior is to use ``self.rtol``.

        Returns:
            True if operators are equivalent, False if not.

        Raises:
            ValueError: Raised if either operator contains parameters
        """
        if not isinstance(other, self.__class__):
            return False

        if self.is_parameterized() or other.is_parameterized():
            raise ValueError("Cannot compare an operator that contains parameters.")

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
        if not isinstance(other, self.__class__):
            return False

        return self._data == other._data

    def __getitem__(self, __k: str) -> _TCoeff:
        """Get the requested element of the ``SparseLabelOp``."""
        return self._data.__getitem__(__k)

    def __len__(self) -> int:
        """Returns the length of the ``SparseLabelOp``."""
        return self._data.__len__()

    def __iter__(self) -> Iterator[str]:
        """An iterator over the keys of the ``SparseLabelOp``."""
        return self._data.__iter__()

    @abstractmethod
    def terms(self) -> Iterator[tuple[list[tuple[str, int]], _TCoeff]]:
        """Provides an iterator analogous to :meth:`items` but with the labels already split into
        pairs of operation characters and indices.

        Yields:
            A tuple with two items; the first one being a list of pairs of the form (char, int)
            where char represents the operation applied to the register and the integer corresponds
            to the register index on which the operator gets applied; the second item of the
            returned tuple is the coefficient of this term.
        """

    @classmethod
    @abstractmethod
    def from_terms(cls, terms: Sequence[tuple[list[tuple[str, int]], _TCoeff]]) -> SparseLabelOp:
        """Constructs a new ``SparseLabelOp`` from a sequence returned by :meth:`.terms`.

        Args:
            terms: a sequence as returned by :meth:`.terms`.

        Returns:
            The constructed operator.
        """

    @abstractmethod
    def _permute_term(
        self, term: list[tuple[str, int]], permutation: Sequence[int]
    ) -> list[tuple[str, int]]:
        """Applies the index permutation to a single label.

        This takes the expanded format of an operator label as generated by :meth:`.terms` and
        applies the provided index permutation to it. See :meth:`.permute_indices` for more details.

        Args:
            term: the expanded format of an operator label.
            permutation: the index permutation.

        Returns:
            The permuted label terms.
        """

    def permute_indices(self, permutation: Sequence[int]) -> SparseLabelOp:
        """Permutes the indices of the operator.

        This method applies the provided index permutation to all labels of this operator. The
        provided permutation must be a sequence of integers whose length is equal to the
        :attr:`register_length` of the operator. The integer at any given index of the sequence
        indicates the new index which that location will be permuted to. For example:

        .. code-block:: python

           op = SparseLabelOp({"+_0 -_1 +_2 -_3": 1.0})
           permuted_op = op.permute_indices([3, 1, 0, 2])
           assert permuted_op == SparseLabelOp({"+_3 -_1 +_0 -_2": 1.0})

        .. warning::

           This permutation utility is very powerful. Be mindful of the implications such a
           permutation might have on other components of the stack. To name an example, the builtin
           two-qubit reduction of the :class:`.ParityMapper` might not yield the expected results
           when used on permuted operator.

        Args:
            permutation: a sequence of integers indicating the permutation to be applied. See above
                for an example.

        Returns:
            A new operator instance with the permuted indices.

        Raises:
            ValueError: if the length of the ``permutation`` argument does not equal
                :attr:`register_length`.
        """
        if len(permutation) != self.register_length:
            raise ValueError(
                f"This operator acts on a register length of {self.register_length} but the "
                f"permutation you provided has length {len(permutation)}."
            )

        new_op = self.from_terms(
            [(self._permute_term(term, permutation), value) for term, value in self.terms()]
        )
        return self._new_instance(new_op._data)

    def __pow__(self, power):
        if power == 0:
            return self.__class__.one()

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
        return self._new_instance({ind: self[ind] for ind in indices})

    def chop(self, atol: float | None = None) -> SparseLabelOp:
        """Chops the real and imaginary parts of the operator coefficients.

        This function separately chops the real and imaginary parts of all coefficients to the
        provided tolerance. Parameters are chopped only if they are exactly zero.

        Args:
            atol: the tolerance to which to chop. If ``None``, :attr:`atol` will be used.

        Returns:
            The chopped operator.
        """
        atol = atol if atol is not None else self.atol

        new_data = {}
        for key, value in self.items():
            if _to_number(value) == 0:
                continue
            if not isinstance(value, ParameterExpression):
                zero_real = cmath.isclose(value.real, 0.0, abs_tol=atol)
                zero_imag = cmath.isclose(value.imag, 0.0, abs_tol=atol)
                if zero_real and zero_imag:
                    continue
                if zero_imag:
                    value = value.real
                elif zero_real:
                    value = value.imag * 1j
            new_data[key] = value

        return self._new_instance(new_data)

    @abstractmethod
    def simplify(self, atol: float | None = None) -> SparseLabelOp:
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
           further simplifications to be implemented.

        This method returns a new operator (the original operator is not modified).

        Args:
            atol: Absolute numerical tolerance. The default behavior is to use ``self.atol``.

        Returns:
            The simplified operator.
        """

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

        Raises:
            ValueError: Operator contains parameters.
        """
        if self.is_parameterized():
            raise ValueError("Cannot compute norm of an operator that contains parameters.")
        return sum(abs(coeff) ** order for coeff in self.values()) ** (1 / order)

    def is_parameterized(self) -> bool:
        """Returns whether the operator contains any parameters."""
        return any(isinstance(coeff, ParameterExpression) for coeff in self.values())

    def assign_parameters(self, parameters: Mapping[ParameterExpression, _TCoeff]) -> SparseLabelOp:
        """Assign parameters to new parameters or values.

        Args:
            parameters: The mapping from parameters to new parameters or values.

        Returns:
            A new operator with the parameters assigned.
        """
        data = {
            key: (
                value.bind(parameters, allow_unknown_parameters=True)
                if isinstance(value, ParameterExpression)
                else value
            )
            for key, value in self._data.items()
        }
        return self._new_instance(data, other=self)

    def round(self, decimals: int = 0) -> SparseLabelOp:
        """Rounds the operator coefficients to a specified number of decimal places.

        Args:
            decimals: the number of decimal places to round coefficients to. By default this
                will round to the nearest integer value.

        Returns:
            The rounded operator.
        """
        new_data = {key: np.around(value, decimals=decimals) for key, value in self.items()}

        return self._new_instance(new_data)

    def is_zero(self, tol: int | None = None) -> bool:
        r"""Returns true if operator length is zero or all coefficients have value zero.

        Args:
            tol: tolerance for checking coefficient values. If this is `None`,
                :attr:`atol` will be used instead.

        Returns:
            If operator length is zero or all coefficients are zero.
        """
        if len(self) == 0:
            return True
        tol = tol if tol is not None else self.atol
        return all(np.isclose(_to_number(val), 0, atol=tol) for val in self._data.values())

    def parameters(self) -> list[ParameterExpression]:
        """Returns a list of the parameters in the operator.

        Returns:
            A list of the parameters in the operator.
        """
        return [coeff for coeff in self.values() if isinstance(coeff, ParameterExpression)]
