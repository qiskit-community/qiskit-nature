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

"""The 2-body electronic integrals."""

from __future__ import annotations

import itertools
from typing import Optional, Union, cast

import numpy as np

from qiskit.tools import parallel_map
from qiskit_nature import QiskitNatureError
from qiskit_nature.settings import settings
from qiskit_nature.operators.second_quantization import FermionicOp

from .electronic_integrals import ElectronicIntegrals
from .one_body_electronic_integrals import OneBodyElectronicIntegrals
from ..bases import ElectronicBasis, ElectronicBasisTransform


class TwoBodyElectronicIntegrals(ElectronicIntegrals):
    """The 2-body electronic integrals."""

    EINSUM_AO_TO_MO = "pqrs,pi,qj,rk,sl->ijkl"
    EINSUM_CHEM_TO_PHYS = "ijkl->ljik"

    _MATRIX_REPRESENTATIONS = ["Alpha-Alpha", "Beta-Alpha", "Beta-Beta", "Alpha-Beta"]

    # TODO: provide symmetry testing functionality?

    def __init__(
        self,
        basis: ElectronicBasis,
        matrices: Union[np.ndarray, tuple[Optional[np.ndarray], ...]],
        threshold: float = ElectronicIntegrals.INTEGRAL_TRUNCATION_LEVEL,
    ) -> None:
        # pylint: disable=line-too-long
        """
        Args:
            basis: the basis which these integrals are stored in. If this is initialized with
                :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis.SO`,
                these integrals will be used *ad verbatim* during the mapping to a
                :class:`~qiskit_nature.operators.second_quantization.SecondQuantizedOp`.
            matrices: the matrices (one or many) storing the actual electronic integrals. If this is
                a single matrix, ``basis`` must be set to
                :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis.SO`.
                Otherwise, this must be a quartet of matrices, the first one being the
                alpha-alpha-spin matrix (which is required), followed by the beta-alpha-spin,
                beta-beta-spin, and alpha-beta-spin matrices (which are optional). The order of
                these matrices follows the standard assigned of quadrants in a plane geometry. If
                any of the latter three matrices are ``None``, the alpha-alpha-spin matrix will be
                used in their place.  However, the final matrix will be replaced by the transpose of
                the second one, if and only if that happens to differ from ``None``.
            threshold: the truncation level below which to treat the integral as zero-valued.
        """
        num_body_terms = 2
        super().__init__(num_body_terms, basis, matrices, threshold)

    def get_matrix(self, index: int = 0) -> np.ndarray:
        # pylint: disable=line-too-long
        """Returns the integral matrix at the requested index.

        When an internal matrix is `None` this method falls back to the alpha-alpha-spin matrix,
        unless the requested index is 3 (the alpha-beta-spin) matrix and the matrix at index 1 (the
        beta-alpha-spin matrix) is not `None`, in which case the transpose of the latter matrix will
        be returned.

        For more details see also
        :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.electronic_integrals.ElectronicIntegrals.get_matrix`

        Args:
            index: the index of the integral matrix to get.

        Returns:
            The requested integral matrix.

        Raises:
            IndexError: when the requested index exceeds the number of internal matrices.
        """
        if self._basis == ElectronicBasis.SO:
            return cast(np.ndarray, self._matrices)

        if index >= len(self._matrices):
            raise IndexError(
                f"The requested index {index} exceeds the number of internal matrices "
                f"{len(self._matrices)}."
            )

        mat = self._matrices[index]
        if mat is None:
            if index == 3:
                mat = self._matrices[0] if self._matrices[1] is None else self._matrices[1].T
            else:
                mat = self._matrices[0]

        return mat

    def transform_basis(self, transform: ElectronicBasisTransform) -> TwoBodyElectronicIntegrals:
        # pylint: disable=line-too-long
        """Transforms the integrals according to the given transform object.

        If the integrals are already in the correct basis, ``self`` is returned.

        Args:
            transform: the transformation object with the integral coefficients.

        Returns:
            The transformed
            :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.ElectronicIntegrals`.

        Raises:
            QiskitNatureError: if the integrals do not match
                :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasisTransform.initial_basis`.
        """
        if self._basis == transform.final_basis:
            return self

        if self._basis != transform.initial_basis:
            raise QiskitNatureError(
                f"The integrals' basis, {self._basis}, does not match the initial basis of the "
                f"transform, {transform.initial_basis}."
            )

        coeff_alpha = transform.coeff_alpha
        coeff_beta = transform.coeff_beta

        alpha_equal_beta = transform.is_alpha_equal_beta()

        coeff_list = [
            (coeff_alpha, coeff_alpha, coeff_alpha, coeff_alpha),
            (coeff_beta, coeff_beta, coeff_alpha, coeff_alpha),
            (coeff_beta, coeff_beta, coeff_beta, coeff_beta),
            (coeff_alpha, coeff_alpha, coeff_beta, coeff_beta),
        ]
        matrices: list[Optional[np.ndarray]] = []
        for idx, (mat, coeffs) in enumerate(zip(self._matrices, coeff_list)):
            if mat is None:
                if alpha_equal_beta:
                    matrices.append(None)
                    continue
                mat = self.get_matrix(idx)
            matrices.append(
                np.einsum(self.EINSUM_AO_TO_MO, mat, *coeffs, optimize=settings.optimize_einsum)
            )

        return TwoBodyElectronicIntegrals(transform.final_basis, tuple(matrices))

    def to_spin(self) -> np.ndarray:
        """Transforms the integrals into the special
        :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis.SO`
        basis.

        Returns:
            A single matrix containing the ``n-body`` integrals in the spin orbital basis.
        """
        if self._basis == ElectronicBasis.SO:
            return self._matrices  # type: ignore

        so_matrix = np.zeros([2 * s for s in self._matrices[0].shape])
        one_indices = (
            (0, 0, 0, 0),  # alpha-alpha-spin
            (0, 1, 1, 0),  # beta-alpha-spin
            (1, 1, 1, 1),  # beta-beta-spin
            (1, 0, 0, 1),  # alpha-beta-spin
        )
        for idx, (ao_mat, one_idx) in enumerate(zip(self._matrices, one_indices)):
            if ao_mat is None:
                ao_mat = self.get_matrix(idx)
            phys_matrix = np.einsum(self.EINSUM_CHEM_TO_PHYS, ao_mat)
            kron = np.zeros((2, 2, 2, 2))
            kron[one_idx] = 1
            so_matrix -= 0.5 * np.kron(kron, phys_matrix)

        return np.where(np.abs(so_matrix) > self._threshold, so_matrix, 0.0)

    @staticmethod
    def _calc_coeffs_with_ops(indices: tuple[int, ...]) -> list[tuple[str, int]]:
        return [("+", indices[0]), ("+", indices[2]), ("-", indices[3]), ("-", indices[1])]

    def compose(
        self, other: ElectronicIntegrals, einsum_subscript: Optional[str] = None
    ) -> OneBodyElectronicIntegrals:
        # pylint: disable=line-too-long
        """Composes these ``TwoBodyElectronicIntegrals`` with an instance of
        :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.OneBodyElectronicIntegrals`.

        This method requires an ``einsum_subscript`` subscript and produces a new instance of
        :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.OneBodyElectronicIntegrals`.

        Args:
            other: an instance of
                :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.OneBodyElectronicIntegrals`.
            einsum_subscript: an additional ``np.einsum`` subscript.

        Returns:
            The resulting
            :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.OneBodyElectronicIntegrals`.

        Raises:
            TypeError: if ``other`` is not an
                :class:`~qiskit_nature.properties.second_quantization.electronic.integrals.OneBodyElectronicIntegrals`
                instance.
            ValueError: if the bases of ``self`` and ``other`` do not match or if ``einsum_subscript`` is
                ``None``.
            NotImplementedError: if the basis of ``self`` is not
            :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis.AO`.
        """
        if einsum_subscript is None:
            raise ValueError(
                "TwoBodyElectronicIntegrals.compose requires an Einsum summation convention "
                "(`einsum_subscript`) in order to evaluate the composition! It may not be `None`."
            )

        if not isinstance(other, OneBodyElectronicIntegrals):
            raise TypeError(
                "TwoBodyElectronicIntegrals.compose expected an `OneBodyElectronicIntegrals` object"
                f" and not one of type {type(other)}."
            )

        if self._basis != other._basis:
            raise ValueError(
                f"The basis of self, {self._basis.value}, does not match the basis of other, "
                f"{other._basis}!"
            )

        if self._basis != ElectronicBasis.AO:
            raise NotImplementedError(
                "TwoBodyElectronicIntegrals.compose is not yet implemented for integrals in a basis"
                " other than the AO basis!"
            )

        eri = self._matrices[0]

        alpha = np.einsum(einsum_subscript, eri, other._matrices[0])
        beta = None
        if other._matrices[1] is not None:
            beta = np.einsum(einsum_subscript, eri, other._matrices[1])

        return OneBodyElectronicIntegrals(self._basis, (alpha, beta), self._threshold)
