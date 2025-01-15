# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
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

import re
import string
from copy import copy
from numbers import Number
from typing import Any, Generator, Sequence, Type, Union, cast

import numpy as np
from qiskit.quantum_info.operators.mixins import TolerancesMixin

import qiskit_nature.optionals as _optionals

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import COO, DOK, GCXS, SparseArray, as_coo, zeros_like
else:

    def as_coo(*args):
        """Empty as_coo function
        Replacement if sparse.as_coo is not present.
        """
        del args

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


def _unpack_args(sequence: Sequence) -> Sequence[ARRAY_TYPE | str]:
    """An internal utility to recursively unpack a sequence of array-like objects or string
    arguments used in some numpy function."""
    seq: list[ARRAY_TYPE | str] = []
    for obj in sequence:
        if isinstance(obj, Tensor):
            seq.append(obj.array)
        elif isinstance(obj, str):
            seq.append(obj)
        elif isinstance(obj, Sequence):
            seq.append(_unpack_args(obj))
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

    def __init__(
        self,
        array: Number | ARRAY_TYPE | Tensor,
        *,
        label_template: str | None = None,
    ) -> None:
        """
        Args:
            array: the wrapped array object. This can be any of the following objects:

                :`numpy.ndarray`: a dense matrix
                :`sparse.SparseArray`: a sparse matrix
                :`numbers.Number`: any numeric singular value
                :`Tensor`: another Tensor whose ``array`` attribute will be extracted
            label_template: the template string used during the translation procedure implemented in
                :meth:`.SparseLabelOp.from_polynomial_tensor`. When ``None``, this will fall back to
                the default template string - see the :attr:`label_template` property for more
                information.
        """
        if isinstance(array, Tensor):
            array = array._array
        elif isinstance(array, Number):
            array = np.array(array)

        self._array: ARRAY_TYPE = array
        self._label_template = label_template

    @property
    def array(self) -> ARRAY_TYPE:
        """Returns the wrapped array object."""
        return self._array

    def __repr__(self) -> str:
        return repr(self._array)

    def __copy__(self) -> Tensor:
        return Tensor(self._array, label_template=self._label_template)

    def __deepcopy__(self, memo) -> Tensor:
        return Tensor(copy(self._array), label_template=self._label_template)

    def __array__(self, dtype=None) -> ARRAY_TYPE:
        """Makes this class behave like a numpy ndarray.

        See also https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array__
        """
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
            context = kwargs.pop("context", None)
            ret = ufunc(*new_inputs, **kwargs)
            if isinstance(ret, (np.ndarray, SparseArray)):
                return self.__array_wrap__(ret, context)
            return ret
        else:
            return NotImplemented

    # pylint: disable=unused-argument
    def __array_function__(self, func, types, args, kwargs):
        """Another numpy function wrapper necessary for interoperability with numpy.

        See also
        https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__
        """
        new_args = _unpack_args(args)
        context = kwargs.pop("context", None)
        ret = func(*new_args, **kwargs)
        if isinstance(ret, (np.ndarray, SparseArray)):
            return self.__array_wrap__(ret, context)
        return ret

    # pylint: disable=unused-argument
    def __array_wrap__(self, array: ARRAY_TYPE, context=None) -> Tensor:
        """Ensures the returned objects of a numpy function are wrapped again by this class.

        See also
        https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_wrap__
        """
        return self.__class__(array, label_template=self._label_template)

    def __getattr__(self, name: str) -> Any:
        array = self.__array__()
        if hasattr(array, name):
            # expose any attribute of the internally wrapped array object
            return getattr(array, name)
        raise AttributeError

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

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the wrapped array object."""
        return self._array.shape

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the wrapped array object."""
        return len(self._array.shape)

    @property
    def label_template(self) -> str:
        r"""The template string used during the translation implemented in
        :meth:`.SparseLabelOp.from_polynomial_tensor`.

        If the ``label_template`` is set to ``None`` (the default value) during initialization of a
        ``Tensor`` instance, this value will be substituted by the internal default template. Its
        value depends on the dimension of the wrapped matrix: it will repeat ``{}_{{}}`` for every
        dimension (independent of its size). This is explained best with an example:

        .. code-block:: python

            print(Tensor(np.eye(4)).label_template)
            # "{}_{{}} {}_{{}}"

            print(Tensor(np.ones((3, 1, 2)).label_template)
            # "{}_{{}} {}_{{}} {}_{{}}"

            print(Tensor(np.ones((2, 2, 2, 2)).label_template)
            # "{}_{{}} {}_{{}} {}_{{}} {}_{{}}"

        The format of this template allows to construct a :class:`.SparseLabelOp` from the
        ``Tensor``\s stored in a :class:`.PolynomialTensor`. This operation is performed when
        calling :meth:`.SparseLabelOp.from_polynomial_tensor`. There, the template is processed
        using the Python string formatter in two steps:

        1. First, the template is formatted using the key under which the ``Tensor`` object was
           stored inside of the :class:`.PolynomialTensor` object. For example:

           .. code-block:: python

               poly = PolynomialTensor(
                  {
                      "+-": Tensor(np.eye(2)),
                      "++--": Tensor(np.ones((2, 2, 2, 2))),
                  }
              )

               # the label_template will get expanded like so:
               for key, tensor in poly.items():
                  sparse_label_template = tensor.label_template.format(*key)
                  print(key, "->", sparse_label_template)

              # "+-" -> "+_{} -_{}"
              # "++--" -> "+_{} +_{} -_{} -_{}"

        2. Next, these templates are used to build the actual labels of the :class:`.SparseLabelOp`
           being constructed. For that, the indices encountered during :meth:`coord_iter` are used
           for example like so:

           .. code-block:: python

              sparse_label_template = "+_{} -_{}"
              for value, index in Tensor(np.eye(2)).coord_iter():
                  sparse_label = sparse_label_template.format(*index)
                  print(sparse_label, value)

              # "+_0 -_0", 1
              # "+_0 -_1", 0
              # "+_1 -_1", 1
              # "+_1 -_0", 0

        Given that you now understand how the ``label_template`` attribute is being used, this
        allows you to modify how the ``Tensor`` objects stored inside a :class:`.PolynomialTensor`
        are processed when they are being translated into a :class:`.SparseLabelOp`.

        Here is a concrete example which enables you to use chemistry-ordered two-body terms:

        .. code-block:: python

           eri_chem = ...  # chemistry-ordered 2-body integrals (a 4-dimensional array)
           tensor = Tensor(eri_chem)
           tensor.label_template = "+_{{0}} +_{{2}} -_{{3}} -_{{1}}"
           poly = PolynomialTensor({"++--": tensor})
           ferm_op_chem = FermionicOp.from_polynomial_tensor(poly)

           # ferm_op_chem is now identical to the following:
           from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering

           eri_phys = to_physicist_ordering(eri_chem)
           poly = PolynomialTensor({"++--": eri_phys})
           ferm_op_phys = FermionicOp.from_polynomial_tensor(poly)

           print(ferm_op_chem.equiv(ferm_op_phys))  # True

        .. note::

           The string formatting in Python is a very powerful tool `[1]`_. Note, that in the use
           case here, we only ever supply positional arguments in the ``.format(...)`` calls which
           means that you cannot use names to identify replacement fields in your label templates.
           However, you can modify their replacement order using numeric values (like shown below).

           Another detail to keep in mind, is that the number of replacement fields may _not_ exceed
           the number of arguments provided to the ``.format(...)`` call. However, the number of
           arguments _can_ exceed the number of replacement fields in your template (this will not
           cause any errors).

        Both of those noted features were actually used in the example provided above:

        1. a custom ordering of the replacement fields in our template string
        2. a smaller number of replacement fields than arguments (because we already hard-coded the
           ``+`` and ``-`` operator strings such that the first expansion to the
           ``sparse_label_template`` only unpacks one set of curly braces but does not actually
           inject anything into the template)

        .. note::

           You could have also used the following template: ``{}_{{0}} {}_{{2}} {}_{{3}} {}_{{1}}``.
           This will work in the same way if the key under which your ``Tensor`` is stored inside of
           the ``PolynomialTensor`` is ``++--``. We did not do this in the example above to show
           that the number of replacement fields can be smaller than the number of arguments
           provided during the formatting step, and to simplify the example a little bit.

           However, if you were to try to use ``+_{0} +_{2} -_{3} -_{1}`` instead, this will **not**
           work as intended because the both string formatting steps are applied unconditionally!
           Thus, this wrong use case would in fact get expanded to ``+_+ +_- -_- -_+`` in the first
           step of the processing leaving no replacement fields to be processed in the second step.

        .. _[1]: https://docs.python.org/3/library/string.html#formatstrings
        """
        if self._label_template is None:
            return " ".join(["{}_{{}}"] * self.ndim)
        return self._label_template

    @label_template.setter
    def label_template(self, label_template: str | None) -> None:
        self._label_template = label_template

    def _reverse_label_template(self, seq: Sequence) -> list:
        """Reverses the effect of a possibly custom ordering of operators in ``label_template``.

        Args:
            seq: a sequence to be re-ordered to reverse the possibly custom ordering.

        Returns:
            The re-ordered sequence.
        """
        formatted = self.label_template.format(*("",) * len(self.shape)).format(*range(len(seq)))
        ordered_ints = [
            (int(label), idx) for idx, label in enumerate(re.findall(r"\d+", formatted))
        ]
        _, permutation = zip(*sorted(ordered_ints))
        return list(seq[idx] for idx in permutation)

    @_optionals.HAS_SPARSE.require_in_call
    def is_sparse(self) -> bool:
        """Returns whether this tensor is sparse."""
        return isinstance(self._array, SparseArray) or self._array.ndim == 0

    def is_dense(self) -> bool:
        """Returns whether this tensor is dense."""
        return isinstance(self._array, np.ndarray)

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

        if isinstance(other, Tensor) and self._label_template != other._label_template:
            return False

        value = Tensor(self)._array
        other_value = Tensor(other)._array

        self_is_sparse = isinstance(value, SparseArray)
        other_is_sparse = isinstance(other_value, SparseArray)

        if self_is_sparse:
            if other_is_sparse:
                if value.ndim != other_value.ndim:
                    return False
                if value.nnz != other_value.nnz:  # type: ignore[union-attr]
                    return False
                if value.size != other_value.size:
                    return False
                diff = value - other_value
                if diff.nnz != 0:
                    return False
                return True
            value = value.todense()  # type: ignore[union-attr]
        elif other_is_sparse:
            other_value = cast(SparseArray, other_value).todense()

        if not np.array_equal(value, other_value):
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

        if isinstance(other, Tensor) and self._label_template != other._label_template:
            return False

        value = Tensor(self)._array
        other_value = Tensor(other)._array

        self_is_sparse = isinstance(value, SparseArray)
        other_is_sparse = isinstance(other_value, SparseArray)

        if self_is_sparse:
            if other_is_sparse:
                if value.ndim != other_value.ndim:
                    return False
                diff = (value - other_value).todense()
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

        if not np.allclose(value, other_value, atol=self.atol, rtol=self.rtol):
            return False

        return True

    def compose(self, other: Tensor, qargs: None = None, front: bool = False) -> Tensor:
        r"""Returns the matrix multiplication with another ``Tensor``.

        Args:
            other: the other ``Tensor``.
            qargs: UNUSED.
            front: If ``True``, composition uses right matrix multiplication, otherwise left
                multiplication is used (the default).

        Returns:
            The tensor resulting from the composition.

        Raises:
            NotImplementedError: when composing ``Tensor`` instances whose :attr:`label_template`
                attributes are not falling back to the default.
        """
        if self._label_template is not None or other._label_template is not None:
            raise NotImplementedError(
                "Composing Tensor objects with label_template attributes other than the default "
                "value of None is not implemented. Instead, construct the desired matrix manually "
                "and wrap it into a new Tensor instance with the desired label_template."
            )

        a = self if front else other
        b = other if front else self

        return a.__class__(np.outer(a._array, b._array).reshape(a.shape + b.shape))

    def tensor(self, other: Tensor) -> Tensor:
        r"""Returns the tensor product with another ``Tensor``.

        Args:
            other: the other Tensor.

        Returns:
            The tensor resulting from the tensor product, :math:`self \otimes other`.

        Raises:
            NotImplementedError: when tensoring ``Tensor`` instances whose :attr:`label_template`
                attributes are not falling back to the default.

        .. note::
            Tensor uses reversed operator ordering to :meth:`expand`.
            For two tensors of the same type ``a.tensor(b) = b.expand(a)``.
        """
        if self._label_template is not None or other._label_template is not None:
            raise NotImplementedError(
                "Tensoring Tensor objects with label_template attributes other than the default "
                "value of None is not implemented. Instead, construct the desired matrix manually "
                "and wrap it into a new Tensor instance with the desired label_template."
            )

        return self._tensor(self, other)

    def expand(self, other: Tensor) -> Tensor:
        r"""Returns the reverse-order tensor product with another ``Tensor``.

        Args:
            other: the other Tensor.

        Returns:
            The tensor resulting from the tensor product, :math:`other \otimes self`.

        Raises:
            NotImplementedError: when expanding ``Tensor`` instances whose :attr:`label_template`
                attributes are not falling back to the default.

        .. note::
            Expand uses reversed operator ordering to :meth:`tensor`.
            For two tensors of the same type ``a.expand(b) = b.tensor(a)``.
        """
        if self._label_template is not None or other._label_template is not None:
            raise NotImplementedError(
                "Expanding Tensor objects with label_template attributes other than the default "
                "value of None is not implemented. Instead, construct the desired matrix manually "
                "and wrap it into a new Tensor instance with the desired label_template."
            )

        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a: Tensor, b: Tensor) -> Tensor:
        # expand a-matrix into upper left sector
        amat = a._array
        adim = len(a.shape)
        aones = np.zeros((2,) * adim)
        aones[(0,) * adim] = 1.0
        amat = np.kron(aones, amat)
        aeinsum = string.ascii_lowercase[:adim] if adim > 0 else ""

        # expand b-matrix into lower right sector
        bmat = b._array
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
            einsum = as_coo(einsum)

        return cls(einsum)

    def coord_iter(self) -> Generator[tuple[Number, tuple[int, ...]], None, None]:
        """Iterates a matrix yielding pairs of values and their coordinates.

        This is best explained with a simple example:

        .. code-block:: python

            for value, index in Tensor(np.arange(4).reshape((2, 2))).coord_iter():
                print(value, index)

            # 0 (0, 0)
            # 1 (0, 1)
            # 2 (1, 0)
            # 3 (1, 1)

        Yields:
            A tuple containing the matrix value and another tuple of integers indicating the
            "coordinate" (or multi-dimensional index) under which said value can be found.
        """
        # PERF: the following matrix unpacking is a performance bottleneck!
        # We could consider using Rust in the future to improve upon this.
        if isinstance(self._array, np.ndarray):
            for index in np.ndindex(*self._array.shape):
                yield self._array[index], index
        else:
            _optionals.HAS_SPARSE.require_now("SparseArray")
            import sparse as sp  # pylint: disable=import-error

            if isinstance(self._array, sp.SparseArray):
                coo = sp.as_coo(self._array)
                for value, *idx in zip(coo.data, *coo.coords):
                    yield value, tuple(idx)
