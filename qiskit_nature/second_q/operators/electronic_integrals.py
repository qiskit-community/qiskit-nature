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

"""A container class for electronic operator coefficients (a.k.a. electronic integrals)."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Number
from typing import cast

import numpy as np

from qiskit.quantum_info.operators.mixins import LinearMixin

from qiskit_nature.exceptions import QiskitNatureError
import qiskit_nature.optionals as _optionals

from .polynomial_tensor import ARRAY_TYPE, PolynomialTensor
from .tensor_ordering import (
    IndexType,
    find_index_order,
    to_physicist_ordering,
)

if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray
else:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass


class ElectronicIntegrals(LinearMixin):
    r"""A container class for electronic operator coefficients (a.k.a. electronic integrals).

    This class contains multiple :class:`qiskit_nature.second_q.operators.PolynomialTensor`
    instances, dealing with the specific case of storing electronic integrals, where the up- and
    down-spin electronic interactions need to be handled separately. These two spins are also
    commonly referred to by :math:`\alpha` and :math:`\beta`, respectively.

    Specifically, this class stores three :class:`~.PolynomialTensor` instances:

    - :attr:`alpha`: which stores the up-spin integrals
    - :attr:`beta`: which stores the down-spin integrals
    - :attr:`beta_alpha`: which stores beta-alpha-spin two-body integrals

    These tensors are subject to some expectations, namely:

    - for ``alpha`` and ``beta`` only the following keys are allowed: ``""``, ``"+-"``, ``"++--"``
    - for ``beta_alpha`` the only allowed key is ``"++--"``
    - the reported ``register_length`` attributes of all non-empty tensors must match

    There are two ways of constructing the ``ElectronicIntegrals``:

    .. code-block:: python

        # assuming you already have your one- and two-body integrals from somewhere
        h1_a, h2_aa, h1_b, h2_bb, h2_ba = ...

        from qiskit_nature.second_q.operators import ElectronicIntegrals, PolynomialTensor

        alpha = PolynomialTensor({"+-": h1_a, "++--": h2_aa})
        beta = PolynomialTensor({"+-": h1_b, "++--": h2_bb})
        beta_alpha = PolynomialTensor({"++--": h2_ba})

        integrals = ElectronicIntegrals(alpha, beta, beta_alpha)

        # alternatively, the following achieves the same effect:
        integrals = ElectronicIntegrals.from_raw_integrals(h1_a, h2_aa, h1_b, h2_bb, h2_ba)

    This class then exposes common mathematical operations performed on these tensors allowing
    simple manipulation of the underlying data structures.

    .. code-block:: python

        # addition
        integrals + integrals

        # scalar multiplication
        2.0 * integrals

    This class will substitute empty ``beta`` and ``beta_alpha`` tensors with the ``alpha`` tensor
    when necessary. For example, this means the following will happen:

    .. code-block:: python

        integrals_pure = ElectronicIntegrals(alpha)
        integrals_mixed = ElectronicIntegrals(alpha, beta, beta_alpha)

        sum = integrals_pure + integrals_mixed
        print(sum.beta.is_empty())  # False
        print(sum.beta_alpha.is_empty())  # False
        print(sum.beta.equiv(alpha + beta))  # True
        print(sum.beta_alpha.equiv(alpha + beta_alpha))  # True

    The same logic holds for other mathematical operations involving multiple ``ElectronicIntegrals``.

    You can add a custom offset to be included in the operator generated from these coefficients
    like so:

    .. code-block:: python

        from qiskit_nature.second_q.operators import PolynomialTensor

        integrals: ElectronicIntegrals

        offset = 2.5
        integrals.alpha += PolynomialTensor({"": offset})
    """

    _VALID_KEYS = {"", "+-", "++--"}

    def __init__(
        self,
        alpha: PolynomialTensor | None = None,
        beta: PolynomialTensor | None = None,
        beta_alpha: PolynomialTensor | None = None,
        *,
        validate: bool = True,
    ) -> None:
        """
        Any ``None``-valued argument will internally be replaced by an empty :class:`~.PolynomialTensor`
        (see also :meth:`qiskit_nature.second_q.operators.PolynomialTensor.empty`).

        Args:
            alpha: the up-spin electronic integrals
            beta: the down-spin electronic integrals
            beta_alpha: the beta-alpha-spin two-body electronic integrals. This may *only* contain
                the ``++--`` key.
            validate: when set to False, no validation will be performed.

        Raises:
            KeyError: if the ``alpha`` tensor contains keys other than ``""``, ``"+-"``, and ``"++--"``.
            KeyError: if the ``beta`` tensor contains keys other than ``""``, ``"+-"``, and ``"++--"``.
            KeyError: if the ``beta_alpha`` tensor contains keys other than ``"++--"``.
            ValueError: if the reported :attr:`~.PolynomialTensor.register_length` attributes of the
                alpha-, beta-, and beta-alpha-spin tensors do not all match.
        """
        self.alpha = alpha
        self.beta = beta
        self.beta_alpha = beta_alpha
        if validate:
            self._validate()

    def _validate(self):
        """Performs internal validation."""
        self._validate_tensor_keys()
        self._validate_register_lengths()

    def _validate_tensor_keys(self):
        """Validates the keys of all internal tensors."""
        if not self.alpha.keys() <= ElectronicIntegrals._VALID_KEYS:
            raise KeyError(
                "The only allowed keys for the alpha-spin tensor are '', '+-', and '++--', but your"
                f" tensor has keys: {self.alpha.keys()}"
            )

        if not self.beta.keys() <= ElectronicIntegrals._VALID_KEYS:
            raise KeyError(
                "The only allowed keys for the beta-spin tensor are '', '+-', and '++--', but your"
                f" tensor has keys: {self.beta.keys()}"
            )

        if not self.beta_alpha.keys() <= {"++--"}:
            raise KeyError(
                "The only allowed key for the beta-alpha-spin tensor is '++--', but your "
                f" tensor has keys: {self.beta_alpha.keys()}"
            )

    def _validate_register_lengths(self):
        """Validates the reported `register_length` attributes of all internal tensors."""
        alpha_len = self.alpha.register_length
        beta_len = self.beta.register_length
        beta_alpha_len = self.beta_alpha.register_length

        if alpha_len is None:
            if beta_len is not None:
                raise ValueError(
                    f"The reported register_length of your beta-spin tensor, {beta_len}, does not "
                    f"match the alpha-spin tensor one, {alpha_len}."
                )
            if beta_alpha_len is not None:
                raise ValueError(
                    f"The reported register_length of your beta-alpha-spin tensor, {beta_alpha_len}"
                    f", does not match the alpha-spin tensor one, {alpha_len}."
                )
        else:
            if beta_len is not None and alpha_len != beta_len:
                raise ValueError(
                    f"The reported register_length of your beta-spin tensor, {beta_len}, does not "
                    f"match the alpha-spin tensor one, {alpha_len}."
                )
            if beta_alpha_len is not None and alpha_len != beta_alpha_len:
                raise ValueError(
                    f"The reported register_length of your beta-alpha-spin tensor, {beta_alpha_len}"
                    f", does not match the alpha-spin tensor one, {alpha_len}."
                )

    @property
    def alpha(self) -> PolynomialTensor:
        """The up-spin electronic integrals."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: PolynomialTensor | None) -> None:
        self._alpha = alpha if alpha is not None else PolynomialTensor.empty()

    @property
    def beta(self) -> PolynomialTensor:
        """The down-spin electronic integrals."""
        return self._beta

    @beta.setter
    def beta(self, beta: PolynomialTensor | None) -> None:
        self._beta = beta if beta is not None else PolynomialTensor.empty()

    @property
    def beta_alpha(self) -> PolynomialTensor:
        """The beta-alpha-spin two-body electronic integrals."""
        return self._beta_alpha

    @beta_alpha.setter
    def beta_alpha(self, beta_alpha: PolynomialTensor | None) -> None:
        if beta_alpha is None:
            self._beta_alpha = PolynomialTensor.empty()
        else:
            keys = set(beta_alpha)
            if keys and keys != {"++--"}:
                raise ValueError(
                    f"The beta_alpha tensor may only contain a `++--` key, not {keys}."
                )
            self._beta_alpha = beta_alpha

    @property
    def alpha_beta(self) -> PolynomialTensor:
        """The alpha-beta-spin two-body electronic integrals.

        These get reconstructed from :attr:`beta_alpha` by transposing in the physicist' ordering
        convention.
        """
        if self.beta_alpha.is_empty():
            return self.beta_alpha

        beta_alpha = cast(ARRAY_TYPE, self.beta_alpha["++--"])
        alpha_beta = np.moveaxis(beta_alpha, (0, 1), (2, 3))
        return PolynomialTensor({"++--": alpha_beta}, validate=False)

    @property
    def one_body(self) -> ElectronicIntegrals:
        """Returns only the one-body integrals."""
        alpha: PolynomialTensor = None
        if "+-" in self.alpha:
            alpha = PolynomialTensor(
                {"+-": self.alpha["+-"]},
                validate=False,
            )
        beta: PolynomialTensor = None
        if "+-" in self.beta:
            beta = PolynomialTensor(
                {"+-": self.beta["+-"]},
                validate=False,
            )
        return self.__class__(alpha, beta)

    @property
    def two_body(self) -> ElectronicIntegrals:
        """Returns only the two-body integrals."""
        alpha: PolynomialTensor = None
        if "++--" in self.alpha:
            alpha = PolynomialTensor(
                {"++--": self.alpha["++--"]},
                validate=False,
            )
        beta: PolynomialTensor = None
        if "++--" in self.beta:
            beta = PolynomialTensor(
                {"++--": self.beta["++--"]},
                validate=False,
            )
        beta_alpha: PolynomialTensor = None
        if "++--" in self.beta_alpha:
            beta_alpha = PolynomialTensor(
                {"++--": self.beta_alpha["++--"]},
                validate=False,
            )
        return self.__class__(alpha, beta, beta_alpha)

    @property
    def register_length(self) -> int | None:
        """The size of the operator that can be generated from these `ElectronicIntegrals`."""
        alpha_length = self.alpha.register_length
        return alpha_length

    def __eq__(self, other: object) -> bool:
        """Check equality of first ElectronicIntegrals with other

        Args:
            other: second ``ElectronicIntegrals`` object to be compared with the first.
        Returns:
            True when ``ElectronicIntegrals`` objects are equal, False when unequal.
        """
        if not isinstance(other, ElectronicIntegrals):
            return False

        if (
            self.alpha == other.alpha
            and self.beta == other.beta
            and self.beta_alpha == other.beta_alpha
        ):
            return True

        return False

    def equiv(self, other: object) -> bool:
        """Check equivalence of first ElectronicIntegrals with other

        Args:
            other: second ``ElectronicIntegrals`` object to be compared with the first.
        Returns:
            True when ``ElectronicIntegrals`` objects are equivalent, False when not.
        """
        if not isinstance(other, ElectronicIntegrals):
            return False

        if (
            self.alpha.equiv(other.alpha)
            and self.beta.equiv(other.beta)
            and self.beta_alpha.equiv(other.beta_alpha)
        ):
            return True

        return False

    def _multiply(self, other: complex) -> ElectronicIntegrals:
        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        return self.__class__(
            cast(PolynomialTensor, other * self.alpha),
            cast(PolynomialTensor, other * self.beta),
            cast(PolynomialTensor, other * self.beta_alpha),
        )

    def _add(self, other: ElectronicIntegrals, qargs=None) -> ElectronicIntegrals:
        if not isinstance(other, ElectronicIntegrals):
            raise TypeError("Incorrect argument type: other should be ElectronicIntegrals")

        # we need to handle beta separately in order to inject alpha where necessary
        beta: PolynomialTensor = None
        beta_self_empty = self.beta.is_empty()
        beta_other_empty = other.beta.is_empty()
        if not (beta_self_empty and beta_other_empty):
            beta_self = self.alpha if beta_self_empty else self.beta
            beta_other = other.alpha if beta_other_empty else other.beta
            beta = beta_self + beta_other

        return self.__class__(
            self.alpha + other.alpha,
            beta,
            self.beta_alpha + other.beta_alpha,
        )

    @classmethod
    def apply(
        cls,
        function: Callable[..., np.ndarray | SparseArray | Number],
        *operands: ElectronicIntegrals,
        validate: bool = True,
    ) -> ElectronicIntegrals:
        """Exposes the :meth:`qiskit_nature.second_q.operators.PolynomialTensor.apply` method.

        This behaves identical to the ``apply`` implementation of the ``PolynomialTensor``, applied
        to the :attr:`alpha`, :attr:`beta`, and :attr:`beta_alpha` attributes of the provided
        ``ElectronicIntegrals`` operands.

        This method is special, because it handles the scenario in which any operand has a non-empty
        :attr:`beta` attribute, in which case the empty-beta attributes of any other operands will
        be filled with :attr:`alpha` attributes of those operands.
        The same applies to the :attr:`beta_alpha` attributes.

        Args:
            function: the function to apply to the internal arrays of the provided operands. This
                function must take numpy (or sparse) arrays as its positional arguments. The number
                of arguments must match the number of provided operands.
            operands: a sequence of ``ElectronicIntegrals`` instances on which to operate.
            validate: when set to False, no validation will be performed.

        Returns:
            A new ``PolynomialTensor``.
        """
        alpha = PolynomialTensor.apply(function, *(op.alpha for op in operands), validate=validate)

        beta: PolynomialTensor = None
        if any(not op.beta.is_empty() for op in operands):
            # If any beta-entry is non-empty, we have to perform this computation.
            # Empty tensors will be populated with their alpha-terms automatically.
            beta = PolynomialTensor.apply(
                function,
                *(op.alpha if op.beta.is_empty() else op.beta for op in operands),
                validate=validate,
            )

        beta_alpha: PolynomialTensor = None
        if all(not op.beta_alpha.is_empty() for op in operands):
            # We can only perform this operation, when all beta_alpha tensors are non-empty.
            beta_alpha = PolynomialTensor.apply(
                function, *(op.beta_alpha for op in operands), validate=validate
            )
        return cls(alpha, beta, beta_alpha, validate=validate)

    @classmethod
    def einsum(
        cls,
        einsum_map: dict[str, tuple[str, ...]],
        *operands: ElectronicIntegrals,
        validate: bool = True,
    ) -> ElectronicIntegrals:
        """Exposes the :meth:`qiskit_nature.second_q.operators.PolynomialTensor.einsum` method.

        This behaves identical to the ``einsum`` implementation of the ``PolynomialTensor``, applied
        to the :attr:`alpha`, :attr:`beta`, and :attr:`beta_alpha` attributes of the provided
        ``ElectronicIntegrals`` operands.

        This method is special, because it handles the scenario in which any operand has a non-empty
        :attr:`beta` attribute, in which case the empty-beta attributes of any other operands will
        be filled with :attr:`alpha` attributes of those operands.
        The same applies to the :attr:`beta_alpha` attributes.

        Args:
            einsum_map: a dictionary, mapping from :meth:`numpy.einsum` subscripts to a tuple of
                strings. These strings correspond to the keys of matrices to be extracted from the
                provided ``ElectronicIntegrals`` operands. The last string in this tuple indicates
                the key under which to store the result in the returned ``ElectronicIntegrals``.
            operands: a sequence of ``ElectronicIntegrals`` instances on which to operate.
            validate: when set to False, no validation will be performed.

        Returns:
            A new ``PolynomialTensor``.
        """
        alpha = PolynomialTensor.einsum(
            einsum_map, *(op.alpha for op in operands), validate=validate
        )

        beta: PolynomialTensor = None
        if any(not op.beta.is_empty() for op in operands):
            # If any beta-entry is non-empty, we have to perform this computation.
            # Empty tensors will be populated with their alpha-terms automatically.
            beta = PolynomialTensor.einsum(
                einsum_map,
                *(op.alpha if op.beta.is_empty() else op.beta for op in operands),
                validate=validate,
            )

        beta_alpha: PolynomialTensor = None
        if all(not op.beta_alpha.is_empty() for op in operands):
            # We can only perform this operation, when all beta_alpha tensors are non-empty.
            beta_alpha = PolynomialTensor.einsum(
                einsum_map, *(op.beta_alpha for op in operands), validate=validate
            )
        return cls(alpha, beta, beta_alpha, validate=validate)

    # pylint: disable=invalid-name
    @classmethod
    def from_raw_integrals(
        cls,
        h1_a: np.ndarray | SparseArray,
        h2_aa: np.ndarray | SparseArray | None = None,
        h1_b: np.ndarray | SparseArray | None = None,
        h2_bb: np.ndarray | SparseArray | None = None,
        h2_ba: np.ndarray | SparseArray | None = None,
        *,
        validate: bool = True,
        auto_index_order: bool = True,
    ) -> ElectronicIntegrals:
        """Loads the provided integral matrices into an ``ElectronicIntegrals`` instance.

        When ``auto_index_order`` is enabled,
        :meth:`qiskit_nature.second_q.operators.tensor_ordering.find_index_order` will be used to
        determine the index ordering of the ``h2_aa`` matrix, based on which the two-body matrices
        will automatically be transformed to the physicist' order, which is required by the
        :class:`qiskit_nature.second_q.operators.PolynomialTensor`.

        Args:
            h1_a: the alpha-spin one-body integrals.
            h2_aa: the alpha-alpha-spin two-body integrals.
            h1_b: the beta-spin one-body integrals.
            h2_bb: the beta-beta-spin two-body integrals.
            h2_ba: the beta-alpha-spin two-body integrals.
            validate: whether or not to validate the integral matrices.
            auto_index_order: whether or not to automatically convert the matrices to physicists'
                order.

        Raises:
            QiskitNatureError: if `auto_index_order=True`, upon encountering an invalid
                :class:`qiskit_nature.second_q.operators.tensor_ordering.IndexType`.

        Returns:
            The resulting ``ElectronicIntegrals``.
        """
        alpha_dict = {"+-": h1_a}

        if h2_aa is not None:
            if auto_index_order:
                index_order = find_index_order(h2_aa)
                if index_order == IndexType.UNKNOWN:
                    raise QiskitNatureError(
                        f"The index ordering of the `h2_aa` argument, {index_order}, is invalid.\n"
                        "Provide the two-body matrices in either chemists' or physicists' order, "
                        "or disable the automatic transformation to enforce these matrices to be "
                        "used (`auto_index_order=False`)."
                    )
                h2_aa = to_physicist_ordering(h2_aa, index_order=index_order)
                if h2_bb is not None:
                    h2_bb = to_physicist_ordering(h2_bb, index_order=index_order)
                if h2_ba is not None:
                    h2_ba = to_physicist_ordering(h2_ba, index_order=index_order)

            alpha_dict["++--"] = h2_aa

        alpha = PolynomialTensor(alpha_dict, validate=validate)

        beta = None
        beta_dict = {}
        if h1_b is not None:
            beta_dict["+-"] = h1_b
        if h2_bb is not None:
            beta_dict["++--"] = h2_bb
        if beta_dict:
            beta = PolynomialTensor(beta_dict, validate=validate)

        beta_alpha = None
        if h2_ba is not None:
            beta_alpha = PolynomialTensor({"++--": h2_ba}, validate=validate)

        return cls(alpha, beta, beta_alpha)

    def second_q_coeffs(self) -> PolynomialTensor:
        """Constructs the total ``PolynomialTensor`` contained the second-quantized coefficients.

        This function constructs the spin-orbital basis tensor as a
        :class:`qiskit_nature.second_q.operators.PolynomialTensor`, by arranging the :attr:`alpha`
        and :attr:`beta` attributes in a block-ordered fashion (up-spin integrals cover the first
        part, down-spin integrals the second part of the resulting register space).

        If the :attr:`beta` and/or :attr:`beta_alpha` attributes are empty, the :attr:`alpha` data
        will be used in their place.

        Returns:
            The ``PolynomialTensor`` representing the entire system.
        """
        beta_empty = self.beta.is_empty()
        beta_alpha_empty = self.beta_alpha.is_empty()

        kron_one_body = np.zeros((2, 2))
        kron_two_body = np.zeros((2, 2, 2, 2))
        kron_tensor = PolynomialTensor(
            {"": cast(Number, 1.0), "+-": kron_one_body, "++--": kron_two_body}
        )

        if beta_empty and beta_alpha_empty:
            kron_one_body[(0, 0)] = 1
            kron_one_body[(1, 1)] = 1
            kron_two_body[(0, 0, 0, 0)] = 0.5
            kron_two_body[(0, 1, 1, 0)] = 0.5
            kron_two_body[(1, 0, 0, 1)] = 0.5
            kron_two_body[(1, 1, 1, 1)] = 0.5

            tensor_blocked_spin_orbitals = PolynomialTensor.apply(np.kron, kron_tensor, self.alpha)
            return tensor_blocked_spin_orbitals

        tensor_blocked_spin_orbitals = PolynomialTensor({})
        # pure alpha spin
        kron_one_body[(0, 0)] = 1
        kron_two_body[(0, 0, 0, 0)] = 0.5
        tensor_blocked_spin_orbitals += PolynomialTensor.apply(np.kron, kron_tensor, self.alpha)
        kron_one_body[(0, 0)] = 0
        kron_two_body[(0, 0, 0, 0)] = 0
        # pure beta spin
        kron_one_body[(1, 1)] = 1
        kron_two_body[(1, 1, 1, 1)] = 0.5
        tensor_blocked_spin_orbitals += PolynomialTensor.apply(np.kron, kron_tensor, self.beta)
        kron_one_body[(1, 1)] = 0
        kron_two_body[(1, 1, 1, 1)] = 0
        # beta_alpha spin
        if not beta_alpha_empty:
            kron_tensor = PolynomialTensor({"++--": kron_two_body})
            kron_two_body[(1, 0, 0, 1)] = 0.5
            tensor_blocked_spin_orbitals += PolynomialTensor.apply(
                np.kron, kron_tensor, self.beta_alpha
            )
            kron_two_body[(1, 0, 0, 1)] = 0
            # extract transposed beta_alpha term
            kron_two_body[(0, 1, 1, 0)] = 0.5
            tensor_blocked_spin_orbitals += PolynomialTensor.apply(
                np.kron, kron_tensor, self.alpha_beta
            )
            kron_two_body[(0, 1, 1, 0)] = 0

        return tensor_blocked_spin_orbitals

    def trace_spin(self) -> PolynomialTensor:
        """Returns a :class:`~.PolynomialTensor` where the spin components have been traced out.

        This will sum the :attr:`alpha` and :attr:`beta` components, tracing out the spin.

        Returns:
            A ``PolynomialTensor`` with the spin traced out.
        """
        beta_empty = self.beta.is_empty()
        beta_alpha_empty = self.beta_alpha.is_empty()

        if beta_empty and beta_alpha_empty:
            return cast(PolynomialTensor, 2.0 * self.alpha)

        one_body = self.one_body
        two_body = self.two_body
        tensor_spin_traced = PolynomialTensor({})
        tensor_spin_traced += one_body.alpha
        tensor_spin_traced += one_body.beta
        tensor_spin_traced += two_body.alpha
        tensor_spin_traced += two_body.beta
        if beta_alpha_empty:
            tensor_spin_traced += two_body.alpha
            tensor_spin_traced += two_body.beta
        else:
            tensor_spin_traced += two_body.beta_alpha
            tensor_spin_traced += two_body.alpha_beta

        return tensor_spin_traced
