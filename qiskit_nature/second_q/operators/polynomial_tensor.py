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

"""PolynomialTensor class"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number
from typing import Iterator, cast

import numpy as np

from qiskit.quantum_info.operators.mixins import (
    LinearMixin,
    AdjointMixin,
    GroupMixin,
    TolerancesMixin,
)

from qiskit_nature.settings import settings


class PolynomialTensor(LinearMixin, AdjointMixin, GroupMixin, TolerancesMixin, Mapping):
    """PolynomialTensor class"""

    def __init__(
        self,
        data: Mapping[str, np.ndarray | Number],
        *,
        validate: bool = True,
    ) -> None:
        """
        Args:
            data: mapping of string-based operator keys to coefficient matrix values.
        Raises:
            ValueError: when length of operator key does not match dimensions of value matrix.
            ValueError: when value matrix does not have consistent dimensions.
            ValueError: when some or all value matrices in ``data`` have different dimensions.
        """
        copy_dict: dict[str, np.ndarray] = {}

        dim: int | None = None

        for key, value in data.items():
            if isinstance(value, Number):
                value = np.asarray(value)

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
        """TODO."""
        for key in self._data:
            if key == "":
                continue
            return cast(np.ndarray, self[key]).shape[0]
        return None

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

    def __getitem__(self, __k: str) -> (np.ndarray | Number):
        """
        Returns value matrix in the `PolynomialTensor`.

        Args:
            __k: operator key string in the `PolynomialTensor`.
        Returns:
            Value matrix corresponding to the operator key `__k`
        """
        return self._data.__getitem__(__k)

    def __len__(self) -> int:
        """
        Returns length of `PolynomialTensor`.
        """
        return self._data.__len__()

    def __iter__(self) -> Iterator[str]:
        """
        Returns iterator of the `PolynomialTensor`.
        """
        return self._data.__iter__()

    def _multiply(self, other: complex) -> PolynomialTensor:
        """Scalar multiplication of PolynomialTensor with complex

        Args:
            other: scalar to be multiplied with the ``PolynomialTensor``.
        Returns:
            the new ``PolynomialTensor`` product object.
        Raises:
            TypeError: if ``other`` is not a ``Number``.
        """
        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        prod_dict: dict[str, np.ndarray] = {}
        for key, matrix in self._data.items():
            prod_dict[key] = np.multiply(matrix, other)
        return PolynomialTensor(prod_dict, validate=False)

    def _add(self, other: PolynomialTensor, qargs=None) -> PolynomialTensor:
        """Addition of PolynomialTensors

        Args:
            other: second ``PolynomialTensor`` object to be added to the first.
        Returns:
            the new summed ``PolynomialTensor``.
        Raises:
            TypeError: when ``other`` is not a ``PolynomialTensor``.
            ValueError: when values corresponding to keys in ``other`` and
                            the first ``PolynomialTensor`` object do not match.
        """
        if not isinstance(other, PolynomialTensor):
            raise TypeError("Incorrect argument type: other should be PolynomialTensor")

        sum_dict = {key: value + other._data.get(key, 0) for key, value in self._data.items()}
        other_unique = {key: other._data[key] for key in other._data.keys() - self._data.keys()}
        sum_dict.update(other_unique)

        return PolynomialTensor(sum_dict, validate=False)

    def __eq__(self, other: object) -> bool:
        """Check equality of first PolynomialTensor with other

        Args:
            other: second ``PolynomialTensor`` object to be compared with the first.
        Returns:
            True when ``PolynomialTensor`` objects are equal, False when unequal.
        """
        if not isinstance(other, PolynomialTensor):
            return False

        if self._data.keys() != other._data.keys():
            return False

        for key, value in self._data.items():
            if not np.array_equal(value, other._data[key]):
                return False
        return True

    def equiv(self, other: object) -> bool:
        """Check equivalence of first PolynomialTensor with other

        Args:
            other: second ``PolynomialTensor`` object to be compared with the first.
        Returns:
            True when ``PolynomialTensor`` objects are equivalent, False when not.
        """
        if not isinstance(other, PolynomialTensor):
            return False

        if self._data.keys() != other._data.keys():
            return False

        for key, value in self._data.items():
            if not np.allclose(value, other._data[key], atol=self.atol, rtol=self.rtol):
                return False
        return True

    def conjugate(self) -> PolynomialTensor:
        """Conjugate of PolynomialTensors

        Returns:
            the complex conjugate of the ``PolynomialTensor``.
        """
        conj_dict: dict[str, np.ndarray] = {}
        for key, value in self._data.items():
            conj_dict[key] = np.conjugate(value)

        return PolynomialTensor(conj_dict, validate=False)

    def transpose(self) -> PolynomialTensor:
        """Transpose of PolynomialTensor

        Returns:
            the transpose of the ``PolynomialTensor``.
        """
        transpose_dict: dict[str, np.ndarray] = {}
        for key, value in self._data.items():
            transpose_dict[key] = np.transpose(value)

        return PolynomialTensor(transpose_dict, validate=False)

    def compose(
        self, other: PolynomialTensor, qargs: None = None, front: bool = False
    ) -> PolynomialTensor:
        r"""Returns the matrix multiplication with another ``PolynomialTensor``.

        Args:
            other: the other PolynomialTensor.
            qargs: UNUSED.
            front: If True composition uses right matrix multiplication, otherwise left
                multiplication is used (the default).

        Returns:
            The operator resulting from the composition.

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
        new_data: dict[str, np.ndarray | Number] = {}
        for key in a:
            if key not in b:
                continue
            if len(key) < 2:
                # fall back to standard multiplication since matmul only applies to at least 2D
                new_data[key] = cast(Number, cast(complex, a[key]) * cast(complex, b[key]))
            else:
                new_data[key] = np.matmul(cast(np.ndarray, a[key]), cast(np.ndarray, b[key]))

        return PolynomialTensor(new_data)

    def tensor(self, other: PolynomialTensor) -> PolynomialTensor:
        r"""Returns the tensor product with another ``PolynomialTensor``.

        Args:
            other: the other PolynomialTensor.

        Returns:
            The tensor resulting from the tensor product, :math:`self \otimes other`.

        .. note::
            The tensor product can be obtained using the ``^`` binary operator.
            Hence ``a.tensor(b)`` is equivalent to ``a ^ b``.

        .. note:
            Tensor uses reversed operator ordering to :meth:`expand`.
            For two tensors of the same type ``a.tensor(b) = b.expand(a)``.
        """
        return self._tensor(self, other)

    def expand(self, other: PolynomialTensor) -> PolynomialTensor:
        r"""Returns the reverse-order tensor product with another ``PolynomialTensor``.

        Args:
            other: the other PolynomialTensor.

        Returns:
            The tensor resulting from the tensor product, :math:`othr \otimes self`.

        .. note:
            Expand is the opposite operator ordering to :meth:`tensor`.
            For two tensors of the same type ``a.expand(b) = b.tensor(a)``.
        """
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a: PolynomialTensor, b: PolynomialTensor) -> PolynomialTensor:
        # NOTE: mypy really does not like Number, so a lot of casts are necessary for the time being
        new_data: dict[str, np.ndarray | Number] = {}
        for key in a:
            if key not in b:
                continue
            if len(key) < 1:
                # handle single value case
                new_data[key] = cast(Number, cast(complex, a[key]) * cast(complex, b[key]))
            else:
                new_data[key] = np.kron(cast(np.ndarray, a[key]), cast(np.ndarray, b[key]))

        return PolynomialTensor(new_data)

    @classmethod
    def einsum(
        cls,
        einsum_map: dict[str, tuple[str, ...]],
        *operands: PolynomialTensor,
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

        Args:
            einsum_map: a dictionary, mapping from :meth:`numpy.einsum` subscripts to a tuple of
                strings. These strings correspond to the keys of matrices to be extracted from the
                provided ``PolynomialTensor`` operands. The last string in this tuple indicates the
                key under which to store the result in the returned ``PolynomialTensor``.
            operands: a sequence of ``PolynomialTensor`` instances on which to operate.

        Returns:
            A new ``PolynomialTensor``.
        """
        new_data: dict[str, np.ndarray] = {}
        for einsum, terms in einsum_map.items():
            *inputs, output = terms
            try:
                result = np.einsum(
                    einsum,
                    *[operands[idx]._data[term] for idx, term in enumerate(inputs)],
                    optimize=settings.optimize_einsum,
                )
            except KeyError:
                continue
            if output in new_data:
                new_data += result
            else:
                new_data[output] = result

        return PolynomialTensor(new_data)
