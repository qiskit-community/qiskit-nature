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

"""The 1-body electronic integrals."""

from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError

from .electronic_integrals import ElectronicIntegrals
from ..bases import ElectronicBasis, ElectronicBasisTransform


class OneBodyElectronicIntegrals(ElectronicIntegrals):
    """The 1-body electronic integrals."""

    def __init__(
        self,
        basis: ElectronicBasis,
        matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]],
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
                Otherwise, this must be a pair of matrices, the first one being the alpha-spin
                matrix (which is required) and the second one being an optional beta-spin matrix. If
                the latter is ``None``, the alpha-spin matrix is used in its place.
            threshold: the truncation level below which to treat the integral as zero-valued.
        """
        num_body_terms = 1
        super().__init__(num_body_terms, basis, matrices, threshold)
        self._matrix_representations = ["Alpha", "Beta"]

    def transform_basis(self, transform: ElectronicBasisTransform) -> "OneBodyElectronicIntegrals":
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

        matrix_a = np.dot(np.dot(transform.coeff_alpha.T, self._matrices[0]), transform.coeff_alpha)
        matrix_b = None
        if self._matrices[1] is not None or not transform.is_alpha_equal_beta():
            matrix_b = np.dot(
                np.dot(transform.coeff_beta.T, self.get_matrix(1)), transform.coeff_beta
            )
        return OneBodyElectronicIntegrals(transform.final_basis, (matrix_a, matrix_b))

    def to_spin(self) -> np.ndarray:
        """Transforms the integrals into the special
        :class:`~qiskit_nature.properties.second_quantization.electronic.bases.ElectronicBasis.SO`
        basis.

        In this case of the 1-body integrals, the returned matrix is a block matrix of the form:
        ``[[alpha_spin, zeros], [zeros, beta_spin]]``.

        Returns:
            A single matrix containing the ``n-body`` integrals in the spin orbital basis.
        """
        if self._basis == ElectronicBasis.SO:
            return self._matrices  # type: ignore

        matrix_a = self.get_matrix(0)
        matrix_b = self.get_matrix(1)
        zeros = np.zeros(matrix_a.shape)
        so_matrix = np.block([[matrix_a, zeros], [zeros, matrix_b]])

        return np.where(np.abs(so_matrix) > self._threshold, so_matrix, 0.0)

    def _calc_coeffs_with_ops(self, indices: Tuple[int, ...]) -> List[Tuple[int, str]]:
        return [(indices[0], "+"), (indices[1], "-")]

    def compose(self, other: ElectronicIntegrals, einsum_subscript: str = "ij,ji") -> complex:
        """Composes these ``OneBodyElectronicIntegrals`` with another instance thereof.

        Args:
            other: an instance of ``OneBodyElectronicIntegrals``.
            einsum_subscript: an additional ``np.einsum`` subscript.

        Returns:
            The resulting complex.

        Raises:
            TypeError: if ``other`` is not an ``OneBodyElectronicIntegrals`` instance.
            ValueError: if the bases of ``self`` and ``other`` do not match.
        """
        if not isinstance(other, OneBodyElectronicIntegrals):
            raise TypeError(
                "OneBodyElectronicIntegrals.compose expected an `OneBodyElectronicIntegrals` object"
                f" and not one of type {type(other)}."
            )

        if self._basis != other._basis:
            raise ValueError(
                f"The basis of self, {self._basis.value}, does not match the basis of other, "
                f"{other._basis}!"
            )

        product = 0.0
        for idx, (front, back) in enumerate(zip(self._matrices, other._matrices)):
            if front is None:
                front = self.get_matrix(idx)
            if back is None:
                back = other.get_matrix(idx)
            product += np.einsum(einsum_subscript, front, back)

        return complex(product)
