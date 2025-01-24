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

# pylint: disable=invalid-name

"""
Symmetric 2-body electronic integrals (:mod:`qiskit_nature.second_q.operators.symmetric_two_body`)
==================================================================================================

.. currentmodule:: qiskit_nature.second_q.operators.symmetric_two_body

This module provides utilities to deal with symmetry-reduced 2-body electronic integrals.

Container classes
-----------------

The classes provided here extend the ``numpy.ndarray`` interface and, thus, may be used as such
interchangeably.

.. note::

   Some operations may not be available on the symmetry-reduced space in which case the instance
   will automatically be unfolded to the full 4-dimensional array. After a successful operation, the
   original symmetry will be attempted to be restored.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SymmetricTwoBodyIntegrals
   S1Integrals
   S4Integrals
   S8Integrals

Unfolding methods
-----------------

These methods can be used to unfold higher symmetries to lower ones.

.. note::

   This implies that the memory consumption **increases**.

.. autosummary::
   :toctree: ../stubs/

   unfold
   unfold_s4_to_s1
   unfold_s8_to_s1
   unfold_s8_to_s4

Folding methods
---------------

These methods can be used to fold lower symmetries to higher ones.

.. note::

   This implies that the memory consumption **decreases**.

.. autosummary::
   :toctree: ../stubs/

   fold
   fold_s1_to_s4
   fold_s1_to_s8
   fold_s4_to_s8

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from itertools import takewhile
from numbers import Number
from typing import Any, Generator, Tuple, cast

import numpy as np

import qiskit_nature.optionals as _optionals

from .tensor import ARRAY_TYPE, Tensor
from .tensor_ordering import IndexType, find_index_order

LOGGER = logging.getLogger(__name__)


def _contracted_indices(maximum: int) -> Generator[int, None, None]:
    for idx in range(maximum):
        yield (idx + 1) * (idx + 2) // 2


def _inflate_index(index: int, maximum: int) -> tuple[int, int]:
    p, q = 0, 0
    for _ in takewhile(lambda pq: pq <= index, _contracted_indices(maximum)):
        p += 1
    q = index - p * (p + 1) // 2
    return p, q


class SymmetricTwoBodyIntegrals(Tensor, ABC):
    """An abstract base class providing the interface for symmetry-reduced two-body electronic
    integral container classes.
    """

    _HANDLED_FUNCTIONS = {np.kron, np.transpose, np.tensordot, np.einsum}

    def __init__(self, eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> None:
        # pylint: disable=unused-argument
        """
        Args:
            eri: the actual two-body tensor.
            validate: when set to ``False``, the requirements of ``eri`` are not validated.

        .. note::

           Actual implementations of this abstract base class may raise a ``ValueError`` if ``eri``
           does not fulfill the requirements imposed by that subclass.
        """
        super().__init__(eri, label_template="+_{{0}} +_{{2}} -_{{3}} -_{{1}}")

    @classmethod
    @abstractmethod
    def zero(cls, norb: int) -> SymmetricTwoBodyIntegrals:
        """Constructs an all-zero integral container of the requested size.

        Args:
            norb: the number of orbitals indicating the dimension of the integral container to be
                returned.

        Returns:
            An integral container of the requested size with all-zero terms.
        """
        raise NotImplementedError()

    def __array_wrap__(self, array: ARRAY_TYPE, context=None) -> Tensor:
        try:
            return self.__class__(array)
        except ValueError:
            # upon construction failure of the SymmetricTwoBodyIntegrals instance we fall back to
            # using a plain Tensor
            return Tensor(array)

    def __array_function__(self, func, types, args, kwargs):
        if func in self._HANDLED_FUNCTIONS:
            return self._numpy_function_via_s1(func, types, args, kwargs)
        try:
            return super().__array_function__(func, types, args, kwargs)
        except np.AxisError:
            return fold(self._numpy_function_via_s1(func, types, args, kwargs))

    def _numpy_function_via_s1(self, func, types, args, kwargs):
        uses_sparse = True
        if func is np.einsum:
            # TODO: figure out why not even opt_einsum can handle this case consistently
            uses_sparse = False
        new_args = []
        for a in args:
            if isinstance(a, SymmetricTwoBodyIntegrals):
                s1 = unfold(a)
                if not uses_sparse:
                    s1 = s1.to_dense()
                new_args.append(s1)
            else:
                new_args.append(a)
        new_types = []
        for t in types:
            if issubclass(t, SymmetricTwoBodyIntegrals):
                new_types.append(S1Integrals)
            else:
                new_types.append(t)
        kwargs["context"] = "_numpy_function_via_s1"
        return super().__array_function__(func, tuple(new_types), tuple(new_args), kwargs)

    def conjugate(self) -> SymmetricTwoBodyIntegrals:
        """Returns the complex conjugate of itself."""
        return self.__class__(self.array.transpose().conjugate())


class S1Integrals(SymmetricTwoBodyIntegrals):
    """A container for 1-fold symmetric 2-body electronic integrals in chemist ordering.

    This class is a utility subclass of the central :class:`.Tensor` used for storing n-dimensional
    arrays. This particular one holds 2-body electronic integrals when unfolded to 1-fold symmetry
    (or in other words the full 4-dimensional array). Even though this provides no reduced memory
    consumption over using a plain array, the benefit of this class is two-fold:

    1. it simplifies the usage of chemist-ordered 2-body electronic integrals in the stack
    2. it can interact with the sibling :class:`.S4Integrals` and :class:`.S8Integrals` classes
    """

    def __init__(self, eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> None:
        """
        Args:
            eri: the 4-dimensional array of the 2-body electronic integrals stored in chemist order.
            validate: when set to ``False``, the requirements of ``eri`` are not validated.

        Raises:
            ValueError: If ``eri`` is not 4-dimensional.
        """
        if validate and len(eri.shape) != 4:
            raise ValueError(
                "Expected a 4-dimensional array but obtained one with shape: ", eri.shape
            )
        super().__init__(eri)

    @classmethod
    def zero(cls, norb: int) -> S1Integrals:
        eri: ARRAY_TYPE
        if _optionals.HAS_SPARSE:
            _optionals.HAS_SPARSE.require_now("DOK")
            import sparse as sp  # pylint: disable=import-error

            eri = sp.DOK((norb, norb, norb, norb))
        else:
            eri = np.zeros((norb, norb, norb, norb))

        return cls(eri)


class S4Integrals(SymmetricTwoBodyIntegrals):
    """A container for 4-fold symmetric 2-body electronic integrals in chemist ordering.

    This class is a utility subclass of the central :class:`.Tensor` used for storing n-dimensional
    arrays. This particular one holds 2-body electronic integrals when contracted to 4-fold
    symmetry. This means, that the full 4-dimensional integrals are contracted to a 2-dimensional
    representation of shape :math:`M x M` where :math:`M` is the number of orbital pairs given by
    :math:`M = N * (N + 1) / 2` where :math:`N` is the number of orbitals.
    """

    def __init__(self, eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> None:
        """
        Args:
            eri: the 2-dimensional array of the 4-fold symmetric 2-body electronic integrals stored
                 in chemist order. The shape of this matrix must be :math:`M x M` where
                 :math:`M = N * (N + 1) / 2` and :math:`N` is the number of orbitals.
            validate: when set to ``False``, the requirements of ``eri`` are not validated.

        Raises:
            ValueError: If ``eri`` is not 2-dimensional.
        """
        if validate and len(eri.shape) != 2:
            raise ValueError(
                "Expected a 2-dimensional array but obtained one with shape: ", eri.shape
            )
        super().__init__(eri)
        self._npair = eri.shape[0]
        self._norb = int(-0.5 + np.sqrt(0.25 + 2 * eri.shape[0]))

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._norb,) * 4

    @classmethod
    def zero(cls, norb: int) -> S4Integrals:
        npair = norb * (norb + 1) // 2
        eri = np.zeros((npair, npair))
        return cls(eri)

    def __array_wrap__(self, array: ARRAY_TYPE, context=None) -> Tensor:
        if len(array.shape) == 4:
            # NOTE: some operations may require implicit unfolding to S1Integrals
            return S1Integrals(array)
        if context is not None and context == "_numpy_function_via_s1":
            return Tensor(array)
        try:
            return self.__class__(array)
        except ValueError:
            # upon construction failure of the SymmetricTwoBodyIntegrals instance we fall back to
            # using a plain Tensor
            return Tensor(array)

    def _reduced_dim_index(self, idx: tuple[int, int, int, int]) -> tuple[int, int]:
        i, j, k, l = idx
        if i < j:
            i, j = j, i
        if k < l:
            k, l = l, k
        ij = i * (i + 1) // 2 + j
        kl = k * (k + 1) // 2 + l
        return (ij, kl)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 4:
            key = cast(Tuple[int, int, int, int], key)
            return self.__array__().__getitem__(self._reduced_dim_index(key))
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Number) -> Any:
        if isinstance(key, tuple) and len(key) == 4:
            key = cast(Tuple[int, int, int, int], key)
            self.array[self._reduced_dim_index(key)] = value
        else:
            self.array[key] = value

    def _full_index(self, idx: tuple[int, int]) -> Generator[tuple[int, int, int, int], None, None]:
        i, j = _inflate_index(idx[0], self._norb)
        k, l = _inflate_index(idx[1], self._norb)
        yield i, j, k, l
        if i > j:
            yield j, i, k, l
        if k > l:
            yield i, j, l, k
        if i > j and k > l:
            yield j, i, l, k

    def coord_iter(self) -> Generator[tuple[Number, tuple[int, ...]], None, None]:
        # PERF: the following matrix unpacking is a performance bottleneck!
        # We could consider using Rust in the future to improve upon this.
        if isinstance(self._array, np.ndarray):
            for index in np.ndindex(*self._array.shape):
                index = cast(Tuple[int, int], index)
                value = self._array[index]
                for full_idx in self._full_index(index):
                    yield value, full_idx
        else:
            _optionals.HAS_SPARSE.require_now("SparseArray")
            import sparse as sp  # pylint: disable=import-error

            if isinstance(self._array, sp.SparseArray):
                coo = sp.as_coo(self._array)
                for value, *idx in zip(coo.data, *coo.coords):
                    index = cast(Tuple[int, int], tuple(idx))
                    for full_idx in self._full_index(index):
                        yield value, tuple(full_idx)


class S8Integrals(SymmetricTwoBodyIntegrals):
    """A container for 8-fold symmetric 2-body electronic integrals in chemist ordering.

    This class is a utility subclass of the central :class:`.Tensor` used for storing n-dimensional
    arrays. This particular one holds 2-body electronic integrals when contracted to 8-fold
    symmetry. This means, that the full 4-dimensional integrals are contracted to a 1-dimensional
    representation of length :math:`M * (M + 1) / 2` where :math:`M` is the number of orbital pairs
    given by :math:`M = N * (N + 1) / 2` where :math:`N` is the number of orbitals.
    """

    def __init__(self, eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> None:
        """
        Args:
            eri: the 1-dimensional array of the 8-fold symmetric 2-body electronic integrals stored
                 in chemist order. The length of this 1D matrix must be :math:`M * (M + 1) / 2`
                 where :math:`M = N * (N + 1) / 2` and :math:`N` is the number of orbitals.
            validate: when set to ``False``, the requirements of ``eri`` are not validated.

        Raises:
            ValueError: If ``eri`` is not 1-dimensional.
        """
        if validate and len(eri.shape) != 1:
            raise ValueError(
                "Expected a 1-dimensional array but obtained one with shape: ", eri.shape
            )
        super().__init__(eri)
        self._npair = int(-0.5 + np.sqrt(0.25 + 2 * eri.shape[0]))
        self._norb = int(-0.5 + np.sqrt(0.25 + 2 * self._npair))

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._norb,) * 4

    @classmethod
    def zero(cls, norb: int) -> S8Integrals:
        npair = norb * (norb + 1) // 2
        eri = np.zeros(npair * (npair + 1) // 2)
        return cls(eri)

    def __array_wrap__(self, array: ARRAY_TYPE, context=None) -> Tensor:
        if len(array.shape) == 4:
            # NOTE: some operations may require implicit unfolding to S1Integrals
            return S1Integrals(array)
        if context is not None and context == "_numpy_function_via_s1":
            return Tensor(array)
        try:
            return self.__class__(array)
        except ValueError:
            # upon construction failure of the SymmetricTwoBodyIntegrals instance we fall back to
            # using a plain Tensor
            return Tensor(array)

    def _reduced_dim_index(self, idx: tuple[int, int, int, int]) -> int:
        i, j, k, l = idx
        if i < j:
            i, j = j, i
        if k < l:
            k, l = l, k
        ij = i * (i + 1) // 2 + j
        kl = k * (k + 1) // 2 + l
        if ij < kl:
            ij, kl = kl, ij
        ijkl = ij * (ij + 1) // 2 + kl
        return ijkl

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 4:
            key = cast(Tuple[int, int, int, int], key)
            return self.__array__().__getitem__(self._reduced_dim_index(key))
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Number) -> Any:
        if isinstance(key, tuple) and len(key) == 4:
            key = cast(Tuple[int, int, int, int], key)
            self.array[self._reduced_dim_index(key)] = value
        else:
            self.array[key] = value

    def _full_index(self, idx: int) -> Generator[tuple[int, int, int, int], None, None]:
        ij, kl = _inflate_index(idx, self._npair)
        i, j = _inflate_index(ij, self._norb)
        k, l = _inflate_index(kl, self._norb)
        yield i, j, k, l
        if i > j:
            yield j, i, k, l
        if k > l:
            yield i, j, l, k
        if i > j and k > l:
            yield j, i, l, k
        if ij > kl:
            i, j, k, l = k, l, i, j
            yield i, j, k, l
            if i > j:
                yield j, i, k, l
            if k > l:
                yield i, j, l, k
            if i > j and k > l:
                yield j, i, l, k

    def coord_iter(self) -> Generator[tuple[Number, tuple[int, ...]], None, None]:
        # PERF: the following matrix unpacking is a performance bottleneck!
        # We could consider using Rust in the future to improve upon this.
        if isinstance(self._array, np.ndarray):
            for index in np.ndindex(*self._array.shape):
                value = self._array[index]
                for full_idx in self._full_index(index[0]):
                    yield value, full_idx
        else:
            _optionals.HAS_SPARSE.require_now("SparseArray")
            import sparse as sp  # pylint: disable=import-error

            if isinstance(self._array, sp.SparseArray):
                coo = sp.as_coo(self._array)
                for value, *idx in zip(coo.data, *coo.coords):
                    for full_idx in self._full_index(idx[0]):
                        yield value, tuple(full_idx)


def unfold(eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> S1Integrals:
    """Unfolds an electronic integrals tensor to 1-fold symmetries (4-dimensional).

    This utility method combines :meth:`.unfold_s4_to_s1` and :meth:`.unfold_s8_to_s1`.

    Args:
        eri: a 4-, 2- or 1-dimensional array storing electronic integrals.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        A 1-fold symmetric tensor.

    Raises:
        NotImplementedError: if ``eri`` is of an unsupported dimension.
    """
    if isinstance(eri, S1Integrals):
        return eri

    if isinstance(eri, Tensor):
        eri = eri.array

    if len(eri.shape) == 2:
        return unfold_s4_to_s1(eri, validate=validate)

    if len(eri.shape) == 1:
        return unfold_s8_to_s1(eri, validate=validate)

    raise NotImplementedError(
        "Expected either a 4-, 2- or 1-dimensional array. Instead, an array of the following shape "
        f"was encountered: {eri.shape}"
    )


def fold(eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> SymmetricTwoBodyIntegrals:
    """Folds an electronic integrals tensor.

    This utility method combines :meth:`.fold_s4_to_s8`, :meth:`.fold_s1_to_s8` and
    :meth:`.fold_s1_to_s4` and attempts to fold the provided tensor as much as possible given
    these folding methods.
    Any ``ValueError`` raised by the methods above is caught. When this happens, this utility will
    try the next folding method.

    Args:
        eri: a 4-, 2- or 1-dimensional array storing electronic integrals.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        Either an instance of :class:`.S8Integrals`, :class:`.S4Integrals` or :class:`.S1Integrals`
        (in this order) depending on the first successful folding method.

    Raises:
        NotImplementedError: if ``eri`` is of an unsupported dimension.
    """
    if isinstance(eri, S8Integrals):
        return eri

    if isinstance(eri, Tensor):
        eri = eri.array

    if len(eri.shape) == 2:
        try:
            return fold_s4_to_s8(eri, validate=validate)
        except ValueError as err:
            LOGGER.warning(
                "The following error was encountered during the attempted conversion of the 4-fold "
                "to 8-fold symmetric integrals: %s",
                err,
            )
            LOGGER.info("Returning 4-fold symmetric integrals as the lowest achievable folding.")
            return S4Integrals(eri, validate=validate)

    if len(eri.shape) == 4:
        try:
            return fold_s1_to_s8(eri, validate=validate)
        except ValueError as err:
            LOGGER.warning(
                "The following error was encountered during the attempted conversion of the 1-fold "
                "to 8-fold symmetric integrals: %s",
                err,
            )
            try:
                return fold_s1_to_s4(eri, validate=validate)
            except ValueError:
                LOGGER.warning(
                    "The following error was encountered during the attempted conversion of the "
                    "1-fold to 4-fold symmetric integrals: %s",
                    err,
                )
                LOGGER.info("No folding was possible. Returing 1-fold symmetric integrals.")
                return S1Integrals(eri, validate=validate)

    raise NotImplementedError(
        "Expected either a 1-, 2- or 4-dimensional array. Instead, an array of the following shape "
        f"was encountered: {eri.shape}"
    )


def _get_norb_and_npair(eri: Tensor | ARRAY_TYPE) -> tuple[int, int]:
    if isinstance(eri, Tensor):
        eri = eri.array

    if len(eri.shape) == 4:
        norb = eri.shape[0]
        npair = norb * (norb + 1) // 2

    elif len(eri.shape) == 2:
        npair = eri.shape[0]
        norb = int(-0.5 + np.sqrt(0.25 + 2 * npair))

    elif len(eri.shape) == 1:
        npair = int(-0.5 + np.sqrt(0.25 + 2 * eri.shape[0]))
        norb = int(-0.5 + np.sqrt(0.25 + 2 * npair))

    return norb, npair


def fold_s1_to_s4(eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> S4Integrals:
    """Folds a 4-dimensional tensor to 4-fold symmetries (2-dimensional).

    Args:
        eri: the 4-dimensional tensor to fold.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        A 4-fold symmetric tensor.

    Raises:
        ValueError: if ``eri`` is not 4-dimensional.
        ValueError: if ``eri`` does not satisfy the permutational symmetries dictated by the
            chemist' convention for the ordering of two-body electronic integrals.
    """
    LOGGER.info("Attempting 1-fold to 4-fold symmetric folding.")
    if isinstance(eri, Tensor):
        eri = eri.array
    if validate and len(eri.shape) != 4:
        raise ValueError(
            "Expected a 4-dimensional array. Instead, an array of the following shape was "
            f"encountered: {eri.shape}"
        )
    index_order = find_index_order(eri)
    if validate and index_order != IndexType.CHEMIST:
        # TODO: relax this to allow complex integrals which only satisfy a subset of the
        # permutations tested in the `find_index_order` method
        raise ValueError(
            "Expected a tensor satisfying the permutational symmetries dictated by the chemist' "
            "convention for two-body electronic integral storage. Instead, the following index "
            f"ordering was determined: {index_order}"
        )

    try:
        from pyscf.ao2mo.addons import restore

        LOGGER.info("Using PySCF's conversion routine")
        norb, _ = _get_norb_and_npair(eri)
        return S4Integrals(restore("4", eri, norb))
    except ImportError:
        pass

    norb, npair = _get_norb_and_npair(eri)
    new_eri = np.zeros((npair, npair))
    for ij, (i, j) in enumerate(zip(*np.tril_indices(norb))):
        for kl, (k, l) in enumerate(zip(*np.tril_indices(norb))):
            new_eri[ij, kl] = eri[i, j, k, l]
    return S4Integrals(new_eri)


def fold_s1_to_s8(eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> S8Integrals:
    """Folds a 4-dimensional tensor to 8-fold symmetries (1-dimensional).

    Args:
        eri: the 4-dimensional tensor to fold.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        An 8-fold symmetric tensor.

    Raises:
        ValueError: if ``eri`` is not 4-dimensional.
        ValueError: if ``eri`` does not satisfy the permutational symmetries dictated by the
            chemist' convention for the ordering of two-body electronic integrals.
    """
    LOGGER.info("Attempting 1-fold to 8-fold symmetric folding.")
    if isinstance(eri, Tensor):
        eri = eri.array
    if validate and len(eri.shape) != 4:
        raise ValueError(
            "Expected a 4-dimensional array. Instead, an array of the following shape was "
            f"encountered: {eri.shape}"
        )
    index_order = find_index_order(eri)
    if validate and index_order != IndexType.CHEMIST:
        # TODO: relax this to allow complex integrals which only satisfy a subset of the
        # permutations tested in the `find_index_order` method
        raise ValueError(
            "Expected a tensor satisfying the permutational symmetries dictated by the chemist' "
            "convention for two-body electronic integral storage. Instead, the following index "
            f"ordering was determined: {index_order}"
        )

    try:
        from pyscf.ao2mo.addons import restore

        LOGGER.info("Using PySCF's conversion routine")
        norb, _ = _get_norb_and_npair(eri)
        return S8Integrals(restore("8", eri, norb))
    except ImportError:
        pass

    norb, npair = _get_norb_and_npair(eri)
    new_eri = np.zeros(npair * (npair + 1) // 2)
    ijkl = 0
    for ij, (i, j) in enumerate(zip(*np.tril_indices(norb))):
        for kl, (k, l) in enumerate(zip(*np.tril_indices(int(i + 1)))):
            if ij >= kl:
                new_eri[ijkl] = eri[i, j, k, l]
                ijkl += 1
    return S8Integrals(new_eri)


def fold_s4_to_s8(eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> S8Integrals:
    """Folds a 2-dimensional tensor to 8-fold symmetries (1-dimensional).

    Args:
        eri: the 2-dimensional tensor to fold.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        An 8-fold symmetric tensor.

    Raises:
        ValueError: if ``eri`` is not 2-dimensional.
        ValueError: if ``eri`` is not a symmetric tensor.
    """
    LOGGER.info("Attempting 4-fold to 8-fold symmetric conversion.")
    if isinstance(eri, Tensor):
        eri = eri.array
    if validate and len(eri.shape) != 2:
        raise ValueError(
            "Expected a 2-dimensional array. Instead, an array of the following shape was "
            f"encountered: {eri.shape}"
        )
    if validate and not np.allclose(eri, eri.T):
        raise ValueError("Expected a symmetric tensor.")

    try:
        from pyscf.ao2mo.addons import restore

        LOGGER.info("Using PySCF's conversion routine")
        norb, _ = _get_norb_and_npair(eri)
        return S8Integrals(restore("8", eri, norb))
    except ImportError:
        pass

    return S8Integrals(eri[np.tril_indices_from(eri)])


def unfold_s8_to_s4(eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> S4Integrals:
    """Unfolds an 8-fold symmetric tensor to 4-fold symmetries (2-dimensional).

    Args:
        eri: the 1-dimensional tensor to unfold.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        A 4-fold symmetric tensor.

    Raises:
        ValueError: if ``eri`` is not 1-dimensional.
    """
    LOGGER.info("Unfolding 8-fold to 4-fold symmetric integrals.")
    try:
        from pyscf.ao2mo.addons import restore

        LOGGER.info("Using PySCF's conversion routine")
        norb, _ = _get_norb_and_npair(eri)
        return S4Integrals(restore("4", eri, norb))
    except ImportError:
        pass

    if isinstance(eri, Tensor):
        eri = eri.array
    if validate and len(eri.shape) != 1:
        raise ValueError(
            "Expected a 1-dimensional array. Instead, an array of the following shape was "
            f"encountered: {eri.shape}"
        )
    _, npair = _get_norb_and_npair(eri)
    new_eri = np.zeros((npair, npair))
    new_eri[np.tril_indices(npair)] = eri
    new_eri[np.triu_indices(npair, k=1)] = new_eri.T[np.triu_indices(npair, k=1)]
    return S4Integrals(new_eri)


def unfold_s4_to_s1(eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> S1Integrals:
    """Unfolds an 4-fold symmetric tensor to 1-fold symmetries (4-dimensional).

    Args:
        eri: the 2-dimensional tensor to unfold.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        A 1-fold symmetric tensor.

    Raises:
        ValueError: if ``eri`` is not 2-dimensional.
    """
    LOGGER.info("Unfolding 4-fold to 1-fold symmetric integrals.")
    try:
        from pyscf.ao2mo.addons import restore

        LOGGER.info("Using PySCF's conversion routine")
        norb, _ = _get_norb_and_npair(eri)
        return S1Integrals(restore("1", eri, norb))
    except ImportError:
        pass

    if isinstance(eri, Tensor):
        eri = eri.array
    if validate and len(eri.shape) != 2:
        raise ValueError(
            "Expected a 2-dimensional array. Instead, an array of the following shape was "
            f"encountered: {eri.shape}"
        )
    norb, _ = _get_norb_and_npair(eri)

    new_eri: ARRAY_TYPE
    is_sparse = False
    if _optionals.HAS_SPARSE:
        _optionals.HAS_SPARSE.require_now("DOK")
        import sparse as sp  # pylint: disable=import-error

        new_eri = sp.DOK((norb, norb, norb, norb))
        is_sparse = True
    else:
        new_eri = np.zeros((norb, norb, norb, norb))

    for ij, (i, j) in enumerate(zip(*np.tril_indices(norb))):
        for kl, (k, l) in enumerate(zip(*np.tril_indices(norb))):
            new_eri[i, j, k, l] = eri[ij, kl]
            new_eri[i, j, l, k] = eri[ij, kl]

            if is_sparse and i > j:
                # NOTE: we cannot slice a sparse array so we need to write the symmetric values here
                new_eri[j, i, k, l] = eri[ij, kl]
                new_eri[j, i, l, k] = eri[ij, kl]

        if not is_sparse and i > j:
            # NOTE: when our array is dense, we avoid a lot of individual writes above and instead
            # exploit the slicing operation here
            new_eri[j, i, :, :] = new_eri[i, j, :, :]

    if is_sparse:
        new_eri = new_eri.to_coo()

    return S1Integrals(new_eri)


def unfold_s8_to_s1(eri: Tensor | ARRAY_TYPE, *, validate: bool = True) -> S1Integrals:
    """Unfolds an 8-fold symmetric tensor to 1-fold symmetries (4-dimensional).

    Args:
        eri: the 1-dimensional tensor to unfold.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        A 1-fold symmetric tensor.

    Raises:
        ValueError: if ``eri`` is not 1-dimensional.
    """
    LOGGER.info("Unfolding 8-fold to 1-fold symmetric integrals.")
    try:
        from pyscf.ao2mo.addons import restore

        LOGGER.info("Using PySCF's conversion routine")
        norb, _ = _get_norb_and_npair(eri)
        return S1Integrals(restore("1", eri, norb))
    except ImportError:
        pass

    if isinstance(eri, Tensor):
        eri = eri.array
    if validate and len(eri.shape) != 1:
        raise ValueError(
            "Expected a 1-dimensional array. Instead, an array of the following shape was "
            f"encountered: {eri.shape}"
        )
    norb, npair = _get_norb_and_npair(eri)

    new_eri: ARRAY_TYPE
    is_sparse = False
    if _optionals.HAS_SPARSE:
        _optionals.HAS_SPARSE.require_now("DOK")
        import sparse as sp  # pylint: disable=import-error

        new_eri = sp.DOK((norb, norb, norb, norb))
        is_sparse = True
    else:
        new_eri = np.zeros((norb, norb, norb, norb))

    for ij, (i, j) in enumerate(zip(*np.tril_indices(norb))):
        row = np.zeros(npair)
        idx = ij * (ij + 1) // 2
        row[:ij] = eri[idx : idx + ij]
        for a in range(ij, npair):
            idx += a
            row[a] = eri[idx]
        idx = ij * (ij + 1) // 2
        for kl, (k, l) in enumerate(zip(*np.tril_indices(norb))):
            if ij <= kl:
                idx += kl
            elif kl > 0:
                idx += 1
            new_eri[i, j, k, l] = row[kl]
            new_eri[i, j, l, k] = row[kl]

            if is_sparse and i > j:
                # NOTE: we cannot slice a sparse array so we need to write the symmetric values here
                new_eri[j, i, k, l] = row[kl]
                new_eri[j, i, l, k] = row[kl]

        if not is_sparse and i > j:
            # NOTE: when our array is dense, we avoid a lot of individual writes above and instead
            # exploit the slicing operation here
            new_eri[j, i, :, :] = new_eri[i, j, :, :]

    if is_sparse:
        new_eri = new_eri.to_coo()

    return S1Integrals(new_eri)
