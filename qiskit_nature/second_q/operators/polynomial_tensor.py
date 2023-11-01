# This code is part of a Qiskit project.
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

"""PolynomialTensor class"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import product
from numbers import Number
from typing import Iterator, Sequence, Type, cast

import numpy as np

from qiskit.quantum_info.operators.mixins import (
    LinearMixin,
    GroupMixin,
    TolerancesMixin,
)

from qiskit_nature.settings import settings
import qiskit_nature.optionals as _optionals
from qiskit_nature.utils import get_einsum

from .tensor import Tensor

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray, COO, DOK, GCXS
else:

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


class PolynomialTensor(LinearMixin, GroupMixin, TolerancesMixin, Mapping):
    """A container class to store arbitrary operator coefficients.

    This class generalizes the storing of operator coefficients in tensor format. Actual operators
    can be extracted from it using the
    :meth:`qiskit_nature.second_q.operators.SparseLabelOp.from_polynomial_tensor` method on the
    respective subclasses of the ``SparseLabelOp``.

    Internally, this class stores tensors as instances of
    :class:`~qiskit_nature.second_q.operators.Tensor`. Refer to its documentation for more details.
    The storage format maps from string keys to these ``Tensor`` objects. By design, **no**
    assumptions are made about the *contents* of the keys. However, the length of each key
    determines the dimension of the tensor which it maps, too. For example (using numpy arrays for
    the sake of simplicity):

    .. code-block:: python

        import numpy as np

        data = {}
        # the empty string, maps to a 0-dimensional matrix, a single number
        data[""] = 1.0
        # a string of length 1, must map to a 1-dimensional array
        data["+"] = np.array([1, 2])
        # a string of length 2, must map to a 2-dimensional array
        data["+-"] = np.array([[1, 2], [3, 4]])
        # ... and so on

    In general, the idea is that each character in a key will be associated with the corresponding
    axis of the tensor, when an operator gets built from the ``PolynomialTensor`` instance. This
    means, that the previous example would expand for example like so:

    .. code-block:: python

        from qiskit_nature.second_q.operators import FermionicOp, PolynomialTensor

        tensor = PolynomialTensor(data)
        operator = FermionicOp.from_polynomial_tensor(tensor)

        print(operator)
        # Fermionic Operator
        # number spin orbitals=2, number terms=7
        #   1.0
        # + 1 * ( +_0 )
        # + 2 * ( +_1 )
        # + 1 * ( +_0 -_0 )
        # + 2 * ( +_0 -_1 )
        # + 3 * ( +_1 -_0 )
        # + 4 * ( +_1 -_1 )

    **Algebra**

    This class supports the following basic arithmetic operations: addition, subtraction, scalar
    multiplication, and operator multiplication.
    For example,

    Addition

    .. code-block:: python

      matrix = np.array([[0, 1], [2, 3]], dtype=float)
      0.5 * PolynomialTensor({"+-": matrix}) + PolynomialTensor({"+-": matrix})

    Operator multiplication

    .. code-block:: python

      tensor = PolynomialTensor({"+-": matrix})
      print(tensor @ tensor)

    Tensor multiplication

    .. code-block:: python

      print(tensor ^ tensor)

    You can also implement more advanced arithmetic via the :meth:`apply` and :meth:`einsum`
    methods.

    .. code-block:: python

      print(PolynomialTensor.apply(np.transpose, tensor))
      print(PolynomialTensor.apply(np.conjugate, 1j * tensor))
      print(PolynomialTensor.apply(np.kron, tensor, tensor))

      print(PolynomialTensor.einsum({"ij,ji": ("+-", "+-", "")}, tensor, tensor))

    **Sparse Arrays**

    Furthermore, since the ``PolynomialTensor`` is building on top of the
    :class:`~qiskit_nature.second_q.operators.Tensor` class it supports both, dense numpy arrays and
    sparse arrays. Since it needs to support more than 2-dimensional arrays, we rely on the
    `sparse <https://sparse.pydata.org/en/stable/index.html>`_ library.

    .. code-block:: python

        import sparse as sp

        sparse_matrix = sp.as_coo(matrix)
        print(PolynomialTensor({"+-": sparse_matrix}))

    One can convert between dense and sparse representation of the same tensor via the
    :meth:`to_dense` and :meth:`to_sparse` methods, respectively.
    """

    def __init__(
        self,
        data: Mapping[str, np.ndarray | SparseArray | complex | Tensor],
        *,
        validate: bool = True,
    ) -> None:
        """
        Args:
            data: mapping of string-based operator keys to coefficient tensor values. If the values
                are not already of type :class:`~qiskit_nature.second_q.operators.Tensor`, they will
                automatically be wrapped into one.
            validate: when set to False the ``data`` will not be validated. Disable this setting
                with care!

        Raises:
            ValueError: when length of operator key does not match dimensions of value matrix.
            ValueError: when value matrix does not have consistent dimensions.
            ValueError: when some or all value matrices in ``data`` have different dimensions.
        """
        copy_dict: dict[str, Tensor] = {}

        dim: int | None = None

        for key, value in data.items():
            if not isinstance(value, Tensor):
                value = Tensor(value)

            if validate and len(value.shape) != len(key):
                raise ValueError(
                    f"Data key {key} of length {len(key)} does not match "
                    f"data value matrix of dimensions {value.shape}"
                )

            dims = set(value.shape)

            if validate and len(dims) > 1:
                raise ValueError(
                    f"For key {key}: dimensions of value matrix are not identical {value.shape}"
                )

            if dim is None and dims:
                # we use the length of the first axis as the dimension of this tensor
                dim = value.shape[0]

            if validate and len(dims) == 1 and dims.pop() != dim:
                raise ValueError(
                    "Dimensions of value matrices in data dictionary do not all agree with each "
                    f"other. The inferred dimension is {dim}, violating the shape {value.shape} of "
                    f"key '{key}'."
                )

            copy_dict[key] = value

        self._data = copy_dict

    @property
    def register_length(self) -> int | None:
        """The size of the operator that can be generated from this ``PolynomialTensor``."""
        for key in self._data:
            if key == "":
                continue
            return self[key].shape[0]
        return None

    def __repr__(self) -> str:
        data_str = f"{dict(self.items())}"

        return "PolynomialTensor(" f"{data_str})"

    def __str__(self) -> str:
        pre = "Polynomial Tensor\n"
        ret = " " + "\n ".join([f'"{label}":\n{str(coeff)}' for label, coeff in self.items()])
        return pre + ret

    @classmethod
    def empty(cls) -> PolynomialTensor:
        """Constructs an empty tensor.

        Returns:
            The empty tensor.
        """
        return PolynomialTensor({})

    def is_empty(self) -> bool:
        """Returns whether this tensor is empty or not."""
        return len(self) == 0

    @_optionals.HAS_SPARSE.require_in_call
    def is_sparse(self) -> bool:
        """Returns whether all matrices in this tensor are sparse."""
        return all(self[key].is_sparse() for key in self if key != "")

    def is_dense(self) -> bool:
        """Returns whether all matrices in this tensor are dense."""
        return all(self[key].is_dense() for key in self if key != "")

    def __getitem__(self, __k: str) -> Tensor:
        """Gets the value from the ``PolynomialTensor``.

        Args:
            __k: operator key string in the ``PolynomialTensor``.

        Returns:
            Value corresponding to the operator key ``__k``.
        """
        item = self._data.__getitem__(__k)
        return item

    def __len__(self) -> int:
        """Returns the length of the ``PolynomialTensor``."""
        return self._data.__len__()

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator of the ``PolynomialTensor``."""
        return self._data.__iter__()

    def to_dense(self) -> PolynomialTensor:
        """Returns a new instance where all matrices are now dense tensors.

        If the instance on which this method was called already fulfilled this requirement, it is
        returned unchanged.
        """
        if self.is_dense():
            return self

        _optionals.HAS_SPARSE.require_now("SparseArray")
        dense_dict: dict[str, Tensor] = {}
        for key, value in self._data.items():
            dense_dict[key] = value.to_dense()
        return PolynomialTensor(dense_dict, validate=False)

    # TODO: change the following type-hint if/when SparseArray dictates the existence of from_numpy
    @_optionals.HAS_SPARSE.require_in_call
    def to_sparse(
        self, *, sparse_type: Type[COO] | Type[DOK] | Type[GCXS] = COO
    ) -> PolynomialTensor:
        """Returns a new instance where all matrices are now sparse tensors.

        If the instance on which this method was called already fulfilled this requirement, it is
        returned unchanged.

        Args:
            sparse_type: the type to use for the conversion to sparse matrices. Note, that this will
                only be applied to matrices which were previously dense tensors. Sparse arrays of
                another type will not be explicitly converted.

        Returns:
            A new ``PolynomialTensor`` with all its matrices converted to the requested sparse array
            type.
        """
        if self.is_sparse():
            return self

        sparse_dict: dict[str, Tensor] = {}
        for key, value in self._data.items():
            sparse_dict[key] = value.to_sparse(sparse_type=sparse_type)

        return PolynomialTensor(sparse_dict, validate=False)

    def _multiply(self, other: complex) -> PolynomialTensor:
        """Scalar multiplication of a PolynomialTensor with a scalar.

        Args:
            other: scalar to be multiplied with the ``PolynomialTensor``.

        Returns:
            The new ``PolynomialTensor`` product object.

        Raises:
            TypeError: if ``other`` is not a number.
        """
        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        prod_dict: dict[str, Tensor] = {}
        for key, matrix in self._data.items():
            prod_dict[key] = other * matrix

        return PolynomialTensor(prod_dict, validate=False)

    def _add(self, other: PolynomialTensor, qargs=None) -> PolynomialTensor:
        """Addition of ``PolynomialTensor`` instances.

        Args:
            other: second ``PolynomialTensor`` object to be added to the first.

        Returns:
            The new summed ``PolynomialTensor``.

        Raises:
            TypeError: when ``other`` is not a ``PolynomialTensor``.
            ValueError: when values corresponding to keys in ``other`` and the first
                ``PolynomialTensor`` object do not match.
        """
        if not isinstance(other, PolynomialTensor):
            raise TypeError("Incorrect argument type: other should be PolynomialTensor")

        sum_dict = {key: value + other._data.get(key, 0) for key, value in self._data.items()}
        other_unique = {key: other._data[key] for key in other._data.keys() - self._data.keys()}
        sum_dict.update(other_unique)

        return PolynomialTensor(sum_dict, validate=False)

    def __eq__(self, other: object) -> bool:
        """Check equality of ``PolynomialTensor`` instances.

        .. note::
            This check only asserts the internal matrix elements for equality but ignores the type
            of the matrices. As such, it will indicate equality of two ``PolynomialTensor``
            instances even if one contains sparse and the other dense numpy arrays, as long as their
            elements are identical.

        Args:
            other: the second ``PolynomialTensor`` object to be compared with the first.

        Returns:
            True when the ``PolynomialTensor`` objects are equal, False when unequal.
        """
        if not isinstance(other, PolynomialTensor):
            return False

        if self._data.keys() != other._data.keys():
            return False

        for key, value in self._data.items():
            other_value = other._data[key]

            if value != other_value:
                return False

        return True

    def equiv(self, other: object) -> bool:
        """Check equivalence of ``PolynomialTensor`` instances.

        .. note::
            This check only asserts the internal matrix elements for equivalence but ignores the
            type of the matrices. As such, it will indicate equivalence of two ``PolynomialTensor``
            instances even if one contains sparse and the other dense numpy arrays, as long as their
            elements match.

        Args:
            other: the second ``PolynomialTensor`` object to be compared with the first.

        Returns:
            True when the ``PolynomialTensor`` objects are equivalent, False when not.
        """
        if not isinstance(other, PolynomialTensor):
            return False

        if self._data.keys() != other._data.keys():
            return False

        for key, value in self._data.items():
            other_value = other._data[key]

            if not value.equiv(other_value):
                return False

        return True

    def compose(
        self, other: PolynomialTensor, qargs: None = None, front: bool = False
    ) -> PolynomialTensor:
        r"""Returns the matrix multiplication with another ``PolynomialTensor``.

        Args:
            other: the other PolynomialTensor.
            qargs: UNUSED.
            front: If ``True``, composition uses right matrix multiplication, otherwise left
                multiplication is used (the default).

        Raises:
            NotImplementedError: when the two tensors do not have the same :attr:`register_length`.

        Returns:
            The tensor resulting from the composition.

        .. note::
            Composition (``&``) by default is defined as `left` matrix multiplication for operators,
            while ``@`` (equivalent to :meth:`dot`) is defined as `right` matrix multiplication.
            This means that ``A & B == A.compose(B)`` is equivalent to ``B @ A == B.dot(A)`` when
            ``A`` and ``B`` are of the same type.

            Setting the ``front=True`` keyword argument changes this to `right` matrix
            multiplication which is equivalent to the :meth:`dot` method
            ``A.dot(B) == A.compose(B, front=True)``.
        """
        a = self if front else other
        b = other if front else self

        if a.register_length != b.register_length:
            raise NotImplementedError()

        new_data: dict[str, Tensor] = {}
        for akey, bkey in product(a, b):
            new_key = akey + bkey

            atensor = a[akey]
            btensor = b[bkey]

            outer = atensor.compose(btensor, qargs=qargs, front=True)

            if new_key in new_data:
                new_data[new_key] = new_data[new_key] + outer
            else:
                new_data[new_key] = outer

        return PolynomialTensor(new_data)

    def tensor(self, other: PolynomialTensor) -> PolynomialTensor:
        r"""Returns the tensor product with another ``PolynomialTensor``.

        Args:
            other: the other PolynomialTensor.

        Raises:
            NotImplementedError: when the two tensors do not have the same :attr:`register_length`.

        Returns:
            The tensor resulting from the tensor product, :math:`self \otimes other`.

        .. note::
            The tensor product can be obtained using the ``^`` binary operator.
            Hence ``a.tensor(b)`` is equivalent to ``a ^ b``.

        .. note::
            Tensor uses reversed operator ordering to :meth:`expand`.
            For two tensors of the same type ``a.tensor(b) = b.expand(a)``.
        """
        return self._tensor(self, other)

    def expand(self, other: PolynomialTensor) -> PolynomialTensor:
        r"""Returns the reverse-order tensor product with another ``PolynomialTensor``.

        Args:
            other: the other PolynomialTensor.

        Raises:
            NotImplementedError: when the two tensors do not have the same :attr:`register_length`.

        Returns:
            The tensor resulting from the tensor product, :math:`other \otimes self`.

        .. note::
            Expand is the opposite operator ordering to :meth:`tensor`.
            For two tensors of the same type ``a.expand(b) = b.tensor(a)``.
        """
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a: PolynomialTensor, b: PolynomialTensor) -> PolynomialTensor:
        if a.register_length != b.register_length:
            raise NotImplementedError()

        new_data: dict[str, Tensor] = {}
        for akey, bkey in product(a, b):

            atensor = a[akey]
            btensor = b[bkey]

            einsum = atensor.tensor(btensor)

            new_key = akey + bkey
            if new_key in new_data:
                new_data[new_key] = new_data[new_key] + einsum
            else:
                new_data[new_key] = einsum

        return PolynomialTensor(new_data)

    @classmethod
    def apply(
        cls,
        function: Callable[..., np.ndarray | SparseArray | complex],
        *operands: PolynomialTensor,
        multi: bool = False,
        validate: bool = True,
    ) -> PolynomialTensor | list[PolynomialTensor]:
        """Applies the provided function to the common set of keys of the provided tensors.

        The usage of this method is best explained by some examples:

        .. code-block:: python

            import numpy as np
            from qiskit_nature.second_q.opertors import PolynomialTensor
            rand_a = np.random.random((2, 2))
            rand_b = np.random.random((2, 2))
            a = PolynomialTensor({"+-": rand_a})
            b = PolynomialTensor({"+": np.random.random(2), "+-": rand_b})

            # transpose
            a_transpose = PolynomialTensor.apply(np.transpose, a)
            print(a_transpose == PolynomialTensor({"+-": rand_a.transpose()}))  # True

            # conjugate
            a_complex = 1j * a
            a_conjugate = PolynomialTensor.apply(np.conjugate, a_complex)
            print(a_conjugate == PolynomialTensor({"+-": -1j * rand_a}))  # True

            # kronecker product
            ab_kron = PolynomialTensor.apply(np.kron, a, b)
            print(ab_kron == PolynomialTensor({"+-": np.kron(rand_a, rand_b)}))  # True
            # Note: that ab_kron does NOT contain the "+" and "+-+" keys although b contained the
            # "+" key. That is because the function only gets applied to the keys which are common
            # to all tensors passed to it.

            # computing eigenvectors
            hermi_a = np.array([[1, -2j], [2j, 5]])
            a = PolynomialTensor({"+-": hermi_a})
            _, eigenvectors = PolynomialTensor.apply(np.linalg.eigh, a, multi=True, validate=False)
            print(eigenvectors == PolynomialTensor({"+-": np.eigh(hermi_a)[1]}))  # True

        .. note::

            The provided function will only be applied to the internal arrays of the common keys of
            all provided ``PolynomialTensor`` instances. That means, that no cross-products will be
            generated.

        Args:
            function: the function to apply to the internal arrays of the provided operands. This
                function must take numpy (or sparse) arrays as its positional arguments. The number
                of arguments must match the number of provided operands.
            operands: a sequence of ``PolynomialTensor`` instances on which to operate.
            multi: when set to True this indicates that the provided numpy function will return
                multiple new numpy arrays which will each be wrapped into a ``PolynomialTensor``
                instance separately.
            validate: when set to False the ``data`` will not be validated. Disable this setting
                with care!

        Returns:
            A new ``PolynomialTensor`` instance with the resulting arrays.
        """
        common_keys = set.intersection(*(set(op) for op in operands))

        new_tensors: list[dict[str, Tensor]] = [{}]
        for key in common_keys:
            results = cast(Tensor, function(*(op[key] for op in operands)))

            if multi:
                for idx, res in enumerate(results):
                    if idx >= len(new_tensors):
                        new_tensors.append({})
                    new_tensors[idx][key] = res
            else:
                new_tensors[0][key] = results

        if multi:
            return [cls(tensor, validate=validate) for tensor in new_tensors]

        return cls(new_tensors[0], validate=validate)

    @classmethod
    def stack(
        cls,
        function: Callable[..., np.ndarray | SparseArray | Number],
        operands: Sequence[PolynomialTensor],
        *,
        validate: bool = True,
    ) -> PolynomialTensor:
        """Stacks the provided sequence of tensors using the given numpy stacking function.

        The usage of this method is best explained by some examples:

        .. code-block:: python

            import numpy as np
            from qiskit_nature.second_q.opertors import PolynomialTensor
            rand_a = np.random.random((2, 2))
            rand_b = np.random.random((2, 2))
            a = PolynomialTensor({"+-": rand_a})
            b = PolynomialTensor({"+": np.random.random(2), "+-": rand_b})

            # np.hstack
            ab_hstack = PolynomialTensor.stack(np.hstack, [a, b], validate=False)
            print(ab_hstack == PolynomialTensor({"+-": np.hstack([a, b], validate=False)}))  # True

            # np.vstack
            ab_vstack = PolynomialTensor.stack(np.vstack, [a, b], validate=False)
            print(ab_vstack == PolynomialTensor({"+-": np.vstack([a, b], validate=False)}))  # True

        .. note::

            The provided function will only be applied to the internal arrays of the common keys of
            all provided ``PolynomialTensor`` instances. That means, that no cross-products will be
            generated.

        .. note::

            When stacking arrays this will likely lead to array shapes which would fail the shape
            validation check (as you can see from the examples above where we explicitly disable
            them). This is considered an advanced use case which is why the user is left to disable
            this check themselves, to ensure they know what they are doing.

        Args:
            function: the stacking function to apply to the internal arrays of the provided
                operands. This function must take a sequence of numpy (or sparse) arrays as its
                first argument. You should use :code:`functools.partial` if you need to provide
                keyword arguments (e.g. :code:`partial(np.stack, axis=-1)`). Common methods to use
                here are :func:`numpy.hstack` and :func:`numpy.vstack`.
            operands: a sequence of ``PolynomialTensor`` instances on which to operate.
            validate: when set to False the ``data`` will not be validated. Disable this setting
                with care!

        Returns:
            A new ``PolynomialTensor`` instance with the resulting arrays.
        """
        common_keys = set.intersection(*(set(op) for op in operands))
        new_data: dict[str, Tensor | Number] = {}
        for key in common_keys:
            new_data[key] = cast(Tensor, function([*(op[key] for op in operands)]))
        return cls(new_data, validate=validate)

    def split(
        self,
        function: Callable[..., np.ndarray | SparseArray | Number],
        indices_or_sections: int | Sequence[int],
        *,
        validate: bool = True,
    ) -> list[PolynomialTensor]:
        """Splits the acted on tensor instance using the given numpy splitting function.

        The usage of this method is best explained by some examples:

        .. code-block:: python

            import numpy as np
            from qiskit_nature.second_q.opertors import PolynomialTensor
            rand_ab = np.random.random((4, 4))
            ab = PolynomialTensor({"+-": rand_ab})

            # np.hsplit
            a, b = ab.split(np.hsplit, [2], validate=False)
            print(a == PolynomialTensor({"+-": np.hsplit(ab, [2])[0], validate=False)}))  # True
            print(b == PolynomialTensor({"+-": np.hsplit(ab, [2])[1], validate=False)}))  # True

            # np.vsplit
            a, b = ab.split(np.vsplit, [2], validate=False)
            print(a == PolynomialTensor({"+-": np.vsplit(ab, [2])[0], validate=False)}))  # True
            print(b == PolynomialTensor({"+-": np.vsplit(ab, [2])[1], validate=False)}))  # True

        .. note::

            When splitting arrays this will likely lead to array shapes which would fail the shape
            validation check (as you can see from the examples above where we explicitly disable
            them). This is considered an advanced use case which is why the user is left to disable
            this check themselves, to ensure they know what they are doing.

        Args:
            function: the splitting function to use. This function must take a single numpy (or
                sparse) array as its first input followed by a sequence of indices to split on.
                You should use :code:`functools.partial` if you need to provide keyword arguments
                (e.g. :code:`partial(np.split, axis=-1)`). Common methods to use here are
                :func:`numpy.hsplit` and :func:`numpy.vsplit`.
            indices_or_sections: a single index or sequence of indices to split on.
            validate: when set to False the ``data`` will not be validated. Disable this setting
                with care!

        Returns:
            New ``PolynomialTensor`` instances containing the split arrays.
        """
        new_tensors: list[dict[str, Tensor | Number]] = []
        for key, arr in self._data.items():
            for idx, new_arr in enumerate(
                function(arr, indices_or_sections)  # type: ignore[arg-type]
            ):
                if idx < len(new_tensors):
                    new_tensors[idx][key] = new_arr
                else:
                    new_tensors.append({key: new_arr})
        return [self.__class__(new_data, validate=validate) for new_data in new_tensors]

    @classmethod
    def einsum(
        cls,
        einsum_map: dict[str, tuple[str, ...]],
        *operands: PolynomialTensor,
        validate: bool = True,
    ) -> PolynomialTensor:
        """Applies the various Einsum convention operations to the provided tensors.

        This method wraps the :meth:`numpy.einsum` function, allowing very complex operations to be
        applied efficiently to the matrices stored inside the provided ``PolynomialTensor``
        operands.

        As an example, let us compute the exact exchange term of an
        :class:`qiskit_nature.second_q.hamiltonians.ElectronicEnergy` hamiltonian:

        .. code-block:: python

            # a PolynomialTensor containing the two-body terms of an ElectronicEnergy hamiltonian
            two_body = PolynomialTensor({"++--": ...})

            # an electronic density:
            density = PolynomialTensor({"+-": ...})

            # computes the ElectronicEnergy.exchange operator
            exchange = PolynomialTensor.einsum(
                {"pqrs,qs->pr": ("++--", "+-", "+-")},
                two_body,
                density,
            )
            # result will be contained in exchange["+-"]

        Another example is the mapping from the AO to MO basis, as implemented by the
        :class:`qiskit_nature.second_q.transformers.BasisTransformer`.

        .. code-block:: python

            # the one- and two-body integrals of a hamiltonian
            hamiltonian = PolynomialTensor({"+-": ..., "++--": ...})

            # the AO-to-MO transformation coefficients
            mo_coeff = PolynomialTensor({"+-": ...})

            einsum_map = {
                "jk,ji,kl->il": ("+-", "+-", "+-", "+-"),
                "prsq,pi,qj,rk,sl->iklj": ("++--", "+-", "+-", "+-", "+-", "++--"),
            }

            transformed = PolynomialTensor.einsum(
                einsum_map, hamiltonian, mo_coeff, mo_coeff, mo_coeff, mo_coeff
            )
            # results will be contained in transformed["+-"] and transformed["++--"], respectively

        .. note::

           :class:`sparse.SparseArray` supports ``opt_einsum.contract` if ``opt_einsum`` is installed.
           It does not support ``numpy.einsum``. In this case, the resultant
           ``PolynomialTensor`` will contain all dense numpy arrays. If a user would like to work
           with a sparse array instead, they should install ``opt_einsum`` or
           they should convert it explicitly using the :meth:`to_sparse` method.

        Args:
            einsum_map: a dictionary, mapping from :meth:`numpy.einsum` subscripts to a tuple of
                strings. These strings correspond to the keys of matrices to be extracted from the
                provided ``PolynomialTensor`` operands. The last string in this tuple indicates the
                key under which to store the result in the returned ``PolynomialTensor``.
            operands: a sequence of ``PolynomialTensor`` instances on which to operate.
            validate: when set to False the ``data`` will not be validated. Disable this setting
                with care!

        Returns:
            A new ``PolynomialTensor``.
        """
        einsum_func, uses_sparse = get_einsum()
        operand_list = list(operands) if uses_sparse else [op.to_dense() for op in operands]
        new_data: dict[str, Tensor] = {}
        for einsum, terms in einsum_map.items():
            *inputs, output = terms
            try:
                ops = []
                for idx, term in enumerate(inputs):
                    op = operand_list[idx]._data[term]
                    ops.append(op)
                result = einsum_func(einsum, *ops, optimize=settings.optimize_einsum)
            except KeyError:
                continue
            if output in new_data:
                new_data += result
            else:
                new_data[output] = result

        return cls(new_data, validate=validate)
