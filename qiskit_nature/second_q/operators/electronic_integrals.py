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

from numbers import Number
from typing import Iterator, cast

import numpy as np

from qiskit.quantum_info.operators.mixins import AdjointMixin, LinearMixin

from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.settings import settings

from .polynomial_tensor import PolynomialTensor
from .tensor_ordering import (
    IndexType,
    _chem_to_phys,
    find_index_order,
)


class ElectronicIntegrals(AdjointMixin, LinearMixin):
    r"""A container class for electronic operator coefficients (a.k.a. electronic integrals).

    This class contains multiple :class:`qiskit_nature.second_q.operators.PolynomialTensor`
    instances, dealing with the specific case of storing electronic integrals, where the up- and
    down-spin electronic interactions need to be handled separately. These two spins are also
    commonly referred to by :math:`\alpha` and :math:`\beta`, respectively.

    Specifically, this class stores three ``PolynomialTensor`` instances:

    - :attr:`alpha`: which stores the up-spin integrals
    - :attr:`beta`: which stores the down-spin integrals
    - :attr:`mixed`: which stores mixed-spin multi-body integrals

    It exposes common mathematical operations performed on these tensors allowing simple
    manipulation of the underlying data structures.

    .. code-block:: python

        # assuming, you have your one- and two-body integrals from somewhere
        h1_a, h2_aa, h1_b, h2_bb, h2_ba = ...

        from qiskit_nature.second_q.operators import ElectronicIntegrals, PolynomialTensor

        alpha = PolynomialTensor({"+-": h1_a, "++--": h2_aa})
        beta = PolynomialTensor({"+-": h1_b, "++--": h2_bb})
        mixed = PolynomialTensor({"++--": h2_ba})

        integrals = ElectronicIntegrals(alpha, beta, mixed)

        # addition
        integrals + integrals

        # scalar multiplication
        2.0 * integrals

        # conjugation, transposition, adjoint
        integrals.conjugate()
        integrals.transpose()
        integrals.adjoint()
    """

    def __init__(
        self,
        alpha: PolynomialTensor | None = None,
        beta: PolynomialTensor | None = None,
        mixed: PolynomialTensor | None = None,
    ) -> None:
        """
        Any ``None``-valued argument will internally be replaced by an empty ``PolynomialTensor``
        (see also :meth:`qiskit_nature.second_q.operators.PolynomialTensor.empty`).

        Args:
            alpha: the up-spin electronic integrals
            beta: the down-spin electronic integrals
            mixed: the mixed-spin multi-body electronic integrals
        """
        self.alpha = alpha
        self.beta = beta
        self.mixed = mixed

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
    def mixed(self) -> PolynomialTensor:
        """The mixed-spin multi-body electronic integrals."""
        return self._mixed

    @mixed.setter
    def mixed(self, mixed: PolynomialTensor | None) -> None:
        self._mixed = mixed if mixed is not None else PolynomialTensor.empty()

    @property
    def beta_alpha(self) -> PolynomialTensor:
        """The beta-alpha-spin two-body electronic integrals.

        This extracts specifically the two-body term from :attr:`mixed`.
        """
        if "++--" not in self.mixed.keys():
            return self.mixed

        beta_alpha = cast(np.ndarray, self.mixed["++--"])
        return PolynomialTensor({"++--": beta_alpha}, validate=False)

    @property
    def alpha_beta(self) -> PolynomialTensor:
        """The alpha-beta-spin two-body electronic integrals.

        These get reconstructed from :attr:`beta_alpha` by transposing in the physicist' ordering
        convention.
        """
        if "++--" not in self.mixed.keys():
            return self.mixed

        beta_alpha = cast(np.ndarray, self.mixed["++--"])
        alpha_beta = np.einsum("ijkl->klij", beta_alpha, optimize=settings.optimize_einsum)
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
        mixed: PolynomialTensor = None
        if "+-" in self.mixed:
            mixed = PolynomialTensor(
                {"+-": self.mixed["+-"]},
                validate=False,
            )
        return ElectronicIntegrals(alpha, beta, mixed)

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
        mixed: PolynomialTensor = None
        if "++--" in self.mixed:
            mixed = PolynomialTensor(
                {"++--": self.mixed["++--"]},
                validate=False,
            )
        return ElectronicIntegrals(alpha, beta, mixed)

    @property
    def register_length(self) -> int | None:
        """The register length of the internal
        :class:`qiskit_nature.second_q.operators.PolynomialTensor` instances."""
        return self.alpha.register_length

    @register_length.setter
    def register_length(self, reg_length: int | None) -> None:
        self.alpha.register_length = reg_length

    def __getitem__(self, __k: str) -> PolynomialTensor:
        try:
            return self.__getattribute__(__k)
        except AttributeError as exc:
            raise KeyError from exc

    def __iter__(self) -> Iterator[str]:
        for key in "alpha", "beta", "mixed":
            yield key

    def __eq__(self, other: object) -> bool:
        """Check equality of first ElectronicIntegrals with other

        Args:
            other: second ``ElectronicIntegrals`` object to be compared with the first.
        Returns:
            True when ``ElectronicIntegrals`` objects are equal, False when unequal.
        """
        if not isinstance(other, ElectronicIntegrals):
            return False

        if self.alpha == other.alpha and self.beta == other.beta and self.mixed == other.mixed:
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
            and self.mixed.equiv(other.mixed)
        ):
            return True

        return False

    def _multiply(self, other: complex) -> ElectronicIntegrals:
        if not isinstance(other, Number):
            raise TypeError(f"other {other} must be a number")

        return ElectronicIntegrals(
            cast(PolynomialTensor, other * self.alpha),
            cast(PolynomialTensor, other * self.beta),
            cast(PolynomialTensor, other * self.mixed),
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

        return ElectronicIntegrals(
            self.alpha + other.alpha,
            beta,
            self.mixed + other.mixed,
        )

    def conjugate(self) -> ElectronicIntegrals:
        """Complex conjugate of ``ElectronicIntegrals``.

        Returns:
            The complex conjugate of the ``ElectronicIntegrals``.
        """
        return ElectronicIntegrals(
            self.alpha.conjugate(),
            self.beta.conjugate(),
            self.mixed.conjugate(),
        )

    def transpose(self) -> ElectronicIntegrals:
        """Transpose of ``ElectronicIntegrals``.

        Returns:
            The transpose of the ``ElectronicIntegrals``.
        """
        return ElectronicIntegrals(
            self.alpha.transpose(),
            self.beta.transpose(),
            self.mixed.transpose(),
        )

    @classmethod
    def einsum(
        cls,
        einsum_map: dict[str, tuple[str, ...]],
        *operands: ElectronicIntegrals,
    ) -> ElectronicIntegrals:
        """Exposes the :meth:`qiskit_nature.second_q.operators.PolynomialTensor.einsum` method.

        This behaves identical to the einsum implementation of the ``PolynomialTensor``, applied to
        the :attr:`alpha`, :attr:`beta`, and :attr:`mixed` attributes of the provided
        ``ElectronicIntegrals`` operands.

        This method is special, because it handles the scenario in which any operand has a non-empty
        :attr:`beta` attribute, in which case the empty-beta attributes of any other operands will
        be filled with :attr:`alpha` attributes of those operands.
        The same applies to the :attr:`mixed` attributes.

        Args:
            einsum_map: a dictionary, mapping from :meth:`numpy.einsum` subscripts to a tuple of
                strings. These strings correspond to the keys of matrices to be extracted from the
                provided ``ElectronicIntegrals`` operands. The last string in this tuple indicates the
                key under which to store the result in the returned ``ElectronicIntegrals``.
            operands: a sequence of ``ElectronicIntegrals`` instances on which to operate.

        Returns:
            A new ``PolynomialTensor``.
        """
        alpha = PolynomialTensor.einsum(einsum_map, *(op.alpha for op in operands))

        beta: PolynomialTensor = None
        if any(not op.beta.is_empty() for op in operands):
            # If any beta-entry is non-empty, we have to perform this computation.
            # Empty tensors will be populated with their alpha-terms automatically.
            beta = PolynomialTensor.einsum(
                einsum_map,
                *(op.alpha if op.beta.is_empty() else op.beta for op in operands),
            )

        mixed: PolynomialTensor = None
        if all(not op.mixed.is_empty() for op in operands):
            # We can only perform this operation, when all mixed tensors are non-empty.
            mixed = PolynomialTensor.einsum(einsum_map, *(op.mixed for op in operands))
        return ElectronicIntegrals(alpha, beta, mixed)

    # pylint: disable=invalid-name
    @classmethod
    def from_raw_integrals(
        cls,
        h1_a: np.ndarray,
        h2_aa: np.ndarray | None = None,
        h1_b: np.ndarray | None = None,
        h2_bb: np.ndarray | None = None,
        h2_ba: np.ndarray | None = None,
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
                if index_order == IndexType.CHEMIST:
                    h2_aa = _chem_to_phys(h2_aa)
                    h2_bb = _chem_to_phys(h2_bb) if h2_bb is not None else None
                    h2_ba = _chem_to_phys(h2_ba) if h2_ba is not None else None
                elif index_order != IndexType.PHYSICIST:
                    raise QiskitNatureError(
                        f"The index ordering of the `h2_aa` argument, {index_order}, is invalid.\n"
                        "Provide the two-body matrices in either chemists' or physicists' order, "
                        "or disable the automatic transformation to enforce these matrices to be "
                        "used (`auto_index_order=False`)."
                    )

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

        mixed = None
        if h2_ba is not None:
            mixed = PolynomialTensor({"++--": h2_ba}, validate=validate)

        return ElectronicIntegrals(alpha, beta, mixed)

    def second_q_coeffs(self) -> PolynomialTensor:
        """Constructs the total ``PolynomialTensor`` contained the second-quantized coefficients.

        This function constructs a :class:`qiskit_nature.second_q.operators.PolynomialTensor` whose
        size is ``alpha.register_length + beta.register_length``. Effectively, it constructs the
        spin-orbital basis tensor, by arranging the :attr:`alpha` and :attr:`beta` attributes in a
        block-ordered fashion (up-spin integrals cover the first part, down-spin integrals the
        second part of the resulting register space).

        If the :attr:`beta` and/or :attr:`mixed` attributes are empty, the :attr:`alpha` data will
        be used in their place.

        Returns:
            The ``PolynomialTensor`` representing the entire system.
        """
        beta_empty = self.beta.is_empty()
        mixed_empty = self.mixed.is_empty()

        kron_one_body = np.zeros((2, 2))
        kron_two_body = np.zeros((2, 2, 2, 2))
        kron_tensor = PolynomialTensor(
            {"": cast(Number, 1.0), "+-": kron_one_body, "++--": kron_two_body}, register_length=2
        )

        if beta_empty and mixed_empty:
            kron_one_body[(0, 0)] = 1
            kron_one_body[(1, 1)] = 1
            kron_two_body[(0, 0, 0, 0)] = 0.5
            kron_two_body[(0, 1, 1, 0)] = 0.5
            kron_two_body[(1, 0, 0, 1)] = 0.5
            kron_two_body[(1, 1, 1, 1)] = 0.5

            tensor_blocked_spin_orbitals = kron_tensor ^ self.alpha
            return tensor_blocked_spin_orbitals

        tensor_blocked_spin_orbitals = PolynomialTensor({})
        # pure alpha spin
        kron_one_body[(0, 0)] = 1
        kron_two_body[(0, 0, 0, 0)] = 0.5
        tensor_blocked_spin_orbitals += kron_tensor ^ self.alpha
        kron_one_body[(0, 0)] = 0
        kron_two_body[(0, 0, 0, 0)] = 0
        # pure beta spin
        kron_one_body[(1, 1)] = 1
        kron_two_body[(1, 1, 1, 1)] = 0.5
        tensor_blocked_spin_orbitals += kron_tensor ^ self.beta
        kron_one_body[(1, 1)] = 0
        kron_two_body[(1, 1, 1, 1)] = 0
        # mixed spin
        if not mixed_empty:
            kron_tensor = PolynomialTensor({"++--": kron_two_body}, register_length=2)
            kron_two_body[(1, 0, 0, 1)] = 0.5
            tensor_blocked_spin_orbitals += kron_tensor ^ self.beta_alpha
            kron_two_body[(1, 0, 0, 1)] = 0
            # extract transposed mixed term
            kron_two_body[(0, 1, 1, 0)] = 0.5
            tensor_blocked_spin_orbitals += kron_tensor ^ self.alpha_beta
            kron_two_body[(0, 1, 1, 0)] = 0

        return tensor_blocked_spin_orbitals
