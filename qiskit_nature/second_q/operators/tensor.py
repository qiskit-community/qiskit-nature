# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tensor class"""

from __future__ import annotations

import math
import string
from numbers import Number
from typing import Any, Sequence, Type, Union, cast

import numpy as np
from qiskit.quantum_info.operators.mixins import TolerancesMixin

import qiskit_nature.optionals as _optionals

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import COO, DOK, GCXS, SparseArray, zeros_like
else:

    def zeros_like(*args):
        """Empty zeros_like function
        Replacement if sparse.zeros_like is not present.
        """
        del args

    class COO:  # type: ignore
        """Empty COO class
        Replacement if sparse.COO is not present.
        """

        pass

    class DOK:  # type: ignore
        """Empty DOK class
        Replacement if sparse.DOK is not present.
        """

        pass

    class GCXS:  # type: ignore
        """Empty GCXS class
        Replacement if sparse.GCXS is not present.
        """

        pass

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


# pylint: disable=invalid-name
ARRAY_TYPE = Union[np.ndarray, SparseArray]


def _scalar_dunder(ufunc):
    def func(self, *args, **kwargs):
        if isinstance(self._array, Number):
            return ufunc(self._array, *args, **kwargs)
        raise TypeError()

    func.__name__ = f"__{ufunc.__name__}__"
    return func


def _unpack_args(sequence: Sequence) -> Sequence[ARRAY_TYPE]:
    """An internal utility to recursively unpack a sequence of array-like objects."""
    seq: list[ARRAY_TYPE] = []
    for obj in sequence:
        if isinstance(obj, Sequence):
            seq.append(_unpack_args(obj))
        elif isinstance(obj, Tensor):
            seq.append(obj.array)
        else:
            seq.append(obj)
    return seq


class Tensor(np.lib.mixins.NDArrayOperatorsMixin, TolerancesMixin):
    """A tensor representation wrapping single numeric values, dense or sparse arrays.

    This class is designed to unify the usage of tensors throughout the stack. It provides a central
    entry point for the handling of arrays enabling seamless interchange of dense and sparse arrays
    in any use-case. This is done by implementing this class as an ``np.ndarray`` container as
    described `here <https://numpy.org/doc/stable/user/basics.dispatch.html>`_. At the same time
    this class can also wrap a ``sparse.SparseArray`` (which in turn implements the ``np.ndarray``
    interface).
    """

    def __init__(self, array: Number | ARRAY_TYPE | Tensor) -> None:
        """
        Args:
            array: the wrapped array object. This can be any of the following objects:

                :`numpy.ndarray`: a dense matrix
                :`sparse.SparseArray`: a sparse matrix
                :`numbers.Number`: any numeric singular value
                :`Tensor`: another Tensor whose ``array`` attribute will be extracted
        """
        if isinstance(array, Tensor):
            array = array._array

        self._array: Number | ARRAY_TYPE = array

    @property
    def array(self) -> Number | ARRAY_TYPE:
        """Returns the wrapped array object."""
        return self._array

    def __repr__(self) -> str:
        return repr(self._array)

    def __array__(self, dtype=None) -> ARRAY_TYPE:
        """Makes this class behave like a numpy ndarray.

        See also https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array__
        """
        if isinstance(self._array, Number):
            return np.asarray(self._array, dtype=dtype)

        if dtype is None:
            return self._array

        return self._array.astype(dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Makes this class work with numpy ufuncs.

        See also
        https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
        """
        if method == "__call__":
            new_inputs = []
            for i in inputs:
                if isinstance(i, self.__class__):
                    new_inputs.append(i._array)
                else:
                    new_inputs.append(i)
            return ufunc(*new_inputs, **kwargs)
        else:
            return NotImplemented

    # pylint: disable=unused-argument
    def __array_function__(self, func, types, args, kwargs):
        """Another numpy function wrapper necessary for interoperability with numpy.

        See also
        https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__
        """
        new_args = _unpack_args(args)
        return self.__array_wrap__(func(*new_args, **kwargs))

    # pylint: disable=unused-argument
    def __array_wrap__(self, array: ARRAY_TYPE, context=None) -> Tensor:
        """Ensures the returned objects of a numpy function are wrapped again by this class.

        See also
        https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_wrap__
        """
        return self.__class__(array)

    def __getattr__(self, name: str) -> Any:
        array = self.__array__()
        if hasattr(array, name):
            # expose any attribute of the internally wrapped array object
            return getattr(array, name)
        return None

    def __getitem__(self, key: Any) -> Any:
        array = self.__array__()
        if hasattr(array, "__getitem__"):
            # expose any item of the internally wrapped array object
            return array.__getitem__(key)
        return None

    def __setitem__(self, key: Any, value: Any) -> None:
        array = self.__array__()
        if hasattr(array, "__setitem__"):
            # expose any item setting of the internally wrapped array object
            array.__setitem__(key, value)

    __int__ = _scalar_dunder(int)
    __float__ = _scalar_dunder(float)
    __complex__ = _scalar_dunder(complex)
    __round__ = _scalar_dunder(round)
    __trunc__ = _scalar_dunder(math.trunc)
    __floor__ = _scalar_dunder(math.floor)
    __ceil__ = _scalar_dunder(math.ceil)

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the wrapped array object.

        If the internal object is a ``Number``, an empty tuple is returned (which is equivalent to
        the result of ``numpy.asarray(number).shape``).
        """
        if isinstance(self._array, Number):
            return ()

        return self._array.shape

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the wrapped array object.

        If the internal object is a ``Number``, 0 is returned (which is equivalent to the result of
        ``numpy.asarray(number).ndim``).
        """
        if isinstance(self._array, Number):
            return 0

        return len(self._array.shape)

    @_optionals.HAS_SPARSE.require_in_call
    def is_sparse(self) -> bool:
        """Returns whether this tensor is sparse."""
        return isinstance(self._array, (SparseArray, Number))

    def is_dense(self) -> bool:
        """Returns whether this tensor is dense."""
        return isinstance(self._array, (np.ndarray, Number))

    # TODO: change the following type-hint if/when SparseArray dictates the existence of from_numpy
    @_optionals.HAS_SPARSE.require_in_call
    def to_sparse(self, *, sparse_type: Type[COO] | Type[DOK] | Type[GCXS] = COO) -> Tensor:
        """Returns a new instance with the internal array converted to a sparse array.

        If the instance on which this method was called already fulfilled this requirement, it is
        returned unchanged.

        Args:
            sparse_type: the type to use for the conversion to sparse array. Note, that this will
            only be applied if the wrapped array object was dense. Converting an already sparse
            array to another sparse type needs to be done explicitly.

        Returns:
            A new ``Tensor`` with the internal array converted to the requested sparse array type.
        """
        if self.is_sparse():
            return self

        return self.__array_wrap__(sparse_type.from_numpy(self._array))

    def to_dense(self) -> Tensor:
        """Returns a new instance with the internal array converted to a dense numpy array.

        If the instance on which this method was called already fulfilled this requirement, it is
        returned unchanged.
        """
        if self.is_dense():
            return self

        return self.__array_wrap__(cast(SparseArray, self._array).todense())

    # pylint: disable=too-many-return-statements
    def __eq__(self, other: object) -> bool:
        """Check equality of ``Tensor`` instances.

        .. note::
            This check only asserts the internal matrix elements for equivalence but ignores the
            type of the matrices. As such, it will indicate equality of two ``Tensor`` instances
            even if one contains sparse and the other dense numpy arrays, as long as their elements
            are identical.

        Args:
            other: the second ``Tensor`` object to be compared with the first.

        Returns:
            True when the ``Tensor`` objects are equal, False when not.
        """
        if not isinstance(other, (Tensor, Number, np.ndarray, SparseArray)):
            return False

        value = Tensor(self)._array
        other_value = Tensor(other)._array

        self_is_sparse = isinstance(value, SparseArray)
        other_is_sparse = isinstance(other_value, SparseArray)

        if self_is_sparse:
            if other_is_sparse:
                if value.ndim != other_value.ndim:  # type: ignore[union-attr]
                    return False
                if value.nnz != other_value.nnz:  # type: ignore[union-attr]
                    return False
                if value.size != other_value.size:  # type: ignore[union-attr]
                    return False
                diff = value - other_value  # type: ignore[operator]
                if diff.nnz != 0:
                    return False
                return True
            value = value.todense()  # type: ignore[union-attr]
        elif other_is_sparse:
            other_value = cast(SparseArray, other_value).todense()

        if not np.array_equal(value, other_value):  # type: ignore[arg-type]
            return False

        return True

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def equiv(self, other: object) -> bool:
        """Check equivalence of ``Tensor`` instances.

        .. note::
            This check only asserts the internal matrix elements for equivalence but ignores the
            type of the matrices. As such, it will indicate equivalence of two ``Tensor`` instances
            even if one contains sparse and the other dense numpy arrays, as long as their elements
            match.

        Args:
            other: the second ``Tensor`` object to be compared with the first.

        Returns:
            True when the ``Tensor`` objects are equivalent, False when not.
        """
        if not isinstance(other, (Tensor, Number, np.ndarray, SparseArray)):
            return False

        value = Tensor(self)._array
        other_value = Tensor(other)._array

        self_is_sparse = isinstance(value, SparseArray)
        other_is_sparse = isinstance(other_value, SparseArray)

        if self_is_sparse:
            if other_is_sparse:
                if value.ndim != other_value.ndim:  # type: ignore[union-attr]
                    return False
                diff = (value - other_value).todense()  # type: ignore[operator]
                if not np.allclose(
                    diff,
                    zeros_like(diff).todense(),
                    atol=self.atol,
                    rtol=self.rtol,
                ):
                    return False
                return True
            value = value.todense()  # type: ignore[union-attr]
        elif other_is_sparse:
            other_value = cast(SparseArray, other_value).todense()

        if not np.allclose(value, other_value, atol=self.atol, rtol=self.rtol):  # type: ignore[arg-type]
            return False

        return True

    def compose(self, other: Tensor, qargs: None = None, front: bool = False) -> Tensor:
        r"""Returns the matrix multiplication with another ``Tensor``.

        Args:
            other: the other Tensor.
            qargs: UNUSED.
            front: If ``True``, composition uses right matrix multiplication, otherwise left
                multiplication is used (the default).

        Returns:
            The tensor resulting from the composition.
        """
        a = self if front else other
        b = other if front else self

        a_is_number = isinstance(a._array, Number)
        b_is_number = isinstance(b._array, Number)

        if a_is_number:
            if b_is_number:
                return a.__class__(a._array * b._array)  # type: ignore[operator]

            result = cast(ARRAY_TYPE, (a._array * b._array)).reshape(b.shape)  # type: ignore[operator]
            return a.__class__(result)
        elif b_is_number:
            result = cast(ARRAY_TYPE, (b._array * a._array)).reshape(a.shape)  # type: ignore[operator]
            return a.__class__(result)

        return a.__class__(
            np.outer(a._array, b._array).reshape(a.shape + b.shape)  # type: ignore[arg-type]
        )

    def tensor(self, other: Tensor) -> Tensor:
        r"""Returns the tensor product with another ``Tensor``.

        Args:
            other: the other Tensor.

        Returns:
            The tensor resulting from the tensor product, :math:`self \otimes other`.

        .. note::
            Tensor uses reversed operator ordering to :meth:`expand`.
            For two tensors of the same type ``a.tensor(b) = b.expand(a)``.
        """
        return self._tensor(self, other)

    def expand(self, other: Tensor) -> Tensor:
        r"""Returns the reverse-order tensor product with another ``Tensor``.

        Args:
            other: the other Tensor.

        Returns:
            The tensor resulting from the tensor product, :math:`other \otimes self`.

        .. note::
            Expand uses reversed operator ordering to :meth:`tensor`.
            For two tensors of the same type ``a.expand(b) = b.tensor(a)``.
        """
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a: Tensor, b: Tensor) -> Tensor:
        # expand a-matrix into upper left sector
        amat = cast(ARRAY_TYPE, a._array)
        adim = len(a.shape)
        aones = np.zeros((2,) * adim)
        aones[(0,) * adim] = 1.0
        amat = np.kron(aones, amat)
        aeinsum = string.ascii_lowercase[:adim] if adim > 0 else ""

        # expand b-matrix into lower right sector
        bmat = cast(ARRAY_TYPE, b._array)
        bdim = len(b.shape)
        bones = np.zeros((2,) * bdim)
        bones[(1,) * bdim] = 1.0
        bmat = np.kron(bones, bmat)
        beinsum = string.ascii_lowercase[-bdim:] if bdim > 0 else ""

        make_sparse = False
        if isinstance(amat, SparseArray):
            # pylint: disable=no-member
            amat = amat.todense()
            make_sparse = True
        if isinstance(bmat, SparseArray):
            # pylint: disable=no-member
            bmat = bmat.todense()
            make_sparse = True
        einsum = np.einsum(f"{aeinsum},{beinsum}", amat, bmat)
        if make_sparse:
            einsum = COO(einsum)

        return cls(einsum)
