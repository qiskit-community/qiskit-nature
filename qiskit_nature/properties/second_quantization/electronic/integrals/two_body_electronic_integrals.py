# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The 2-body electronic integrals."""

from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError

from .electronic_integrals import ElectronicIntegrals
from .one_body_electronic_integrals import OneBodyElectronicIntegrals
from ..bases import ElectronicBasis, ElectronicBasisTransform


class TwoBodyElectronicIntegrals(ElectronicIntegrals):
    """The 2-body electronic integrals."""

    EINSUM_AO_TO_MO = "pqrs,pi,qj,rk,sl->ijkl"
    EINSUM_CHEM_TO_PHYS = "ijkl->ljik"

    # TODO: provide symmetry testing functionality?

    def __init__(
        self,
        basis: ElectronicBasis,
        matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]],
        threshold: float = ElectronicIntegrals.INTEGRAL_TRUNCATION_LEVEL,
    ):
        """
        Args:
            basis: the basis which these integrals are stored in. If this is initialized with
                ``ElectronicBasis.SO``, these integrals will be used *ad verbatim* during the
                mapping to a ``SecondQuantizedOp``.
            matrices: the matrices (one or many) storing the actual electronic integrals. If this is
                a single matrix, ``basis`` must be set to ``ElectronicBasis.SO``. Otherwise, this
                must be a quartet of matrices, the first one being the alpha-alpha-spin matrix
                (which is required), followed by the beta-alpha-spin, beta-beta-spin, and
                alpha-beta-spin matrices (which are optional). The order of these matrices follows
                the standard assigned of quadrants in a plane geometry. If any of the latter three
                matrices are ``None``, the alpha-alpha-spin matrix will be used in their place.
                However, the final matrix will be replaced by the transpose of the second one, if
                and only if that happens to differ from ``None``.
            threshold: the truncation level below which to treat the integral in the SO matrix as
                zero-valued.
        """
        num_body_terms = 2

        filled_matrices = []
        alpha_beta_spin_idx = 3
        for idx, mat in enumerate(matrices):
            if mat is not None:
                filled_matrices.append(mat)
            elif idx == alpha_beta_spin_idx:
                if matrices[1] is None:
                    filled_matrices.append(matrices[0])
                else:
                    filled_matrices.append(matrices[1].T)
            else:
                filled_matrices.append(matrices[0])

        super().__init__(num_body_terms, basis, tuple(filled_matrices), threshold)
        self._matrix_representations = ["Alpha-Alpha", "Alpha-Beta", "Beta-Beta", "Beta-Alpha"]

    def transform_basis(self, transform: ElectronicBasisTransform) -> "TwoBodyElectronicIntegrals":
        """Transforms the integrals according to the given transform object.

        If the integrals are already in the correct basis, ``self`` is returned.

        Args:
            transform: the transformation object with the integral coefficients.

        Returns:
            The transformed ``ElectronicIntegrals``.

        Raises:
            QiskitNatureError: if the integrals do not match
                ``ElectronicBasisTransform.initial_basis``.
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

        coeff_list = [
            (coeff_alpha, coeff_alpha, coeff_alpha, coeff_alpha),
            (coeff_beta, coeff_beta, coeff_alpha, coeff_alpha),
            (coeff_beta, coeff_beta, coeff_beta, coeff_beta),
            (coeff_alpha, coeff_alpha, coeff_beta, coeff_beta),
        ]
        matrices: List[Optional[np.ndarray]] = []
        for mat, coeffs in zip(self._matrices, coeff_list):
            if mat is None:
                matrices.append(None)
                continue
            matrices.append(np.einsum(self.EINSUM_AO_TO_MO, mat, *coeffs))

        return TwoBodyElectronicIntegrals(transform.final_basis, tuple(matrices))

    def to_spin(self) -> np.ndarray:
        """Transforms the integrals into the special ``ElectronicBasis.SO`` basis.

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
        alpha_beta_spin_idx = 3
        for idx, (ao_mat, one_idx) in enumerate(zip(self._matrices, one_indices)):
            if ao_mat is None:
                if idx == alpha_beta_spin_idx:
                    ao_mat = self._matrices[0] if self._matrices[1] is None else self._matrices[1].T
                else:
                    ao_mat = self._matrices[0]
            phys_matrix = np.einsum(self.EINSUM_CHEM_TO_PHYS, ao_mat)
            kron = np.zeros((2, 2, 2, 2))
            kron[one_idx] = 1
            so_matrix -= 0.5 * np.kron(kron, phys_matrix)

        return np.where(np.abs(so_matrix) > self._threshold, so_matrix, 0.0)

    def _calc_coeffs_with_ops(self, indices: Tuple[int, ...]) -> List[Tuple[int, str]]:
        return [(indices[0], "+"), (indices[2], "+"), (indices[3], "-"), (indices[1], "-")]

    def compose(self, other: OneBodyElectronicIntegrals, einsum: str) -> OneBodyElectronicIntegrals:
        """TODO."""
        if not isinstance(other, OneBodyElectronicIntegrals):
            raise TypeError()

        if self._basis != other._basis:
            raise ValueError()

        if self._basis != ElectronicBasis.AO:
            raise NotImplementedError()

        eri = self._matrices[0]

        alpha = np.einsum(einsum, eri, other._matrices[0])
        beta = None
        if other._matrices[1] is not None:
            beta = np.einsum(einsum, eri, other._matrices[1])

        return OneBodyElectronicIntegrals(self._basis, (alpha, beta), self._threshold)
