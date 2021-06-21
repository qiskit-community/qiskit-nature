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
    ) -> None:
        """
        Args:
            basis: the basis which these integrals are stored in. If this is initialized with
                ``ElectronicBasis.SO``, these integrals will be used *ad verbatim* during the
                mapping to a ``SecondQuantizedOp``.
            matrices: the matrices (one or many) storing the actual electronic integrals. If this is
                a single matrix, ``basis`` must be set to ``ElectronicBasis.SO``. Otherwise, this
                must be a pair of matrices, the first one being the alpha-spin matrix (which is
                required) and the second one being an optional beta-spin matrix. If the latter is
                ``None``, the alpha-spin matrix is used in its place.
        """
        num_body_terms = 1
        super().__init__(num_body_terms, basis, matrices)

    def transform_basis(self, transform: ElectronicBasisTransform) -> "OneBodyElectronicIntegrals":
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

        matrix_a = np.dot(np.dot(transform.coeff_alpha.T, self._matrices[0]), transform.coeff_alpha)
        matrix_b = None
        if self._matrices[1] is not None:
            matrix_b = np.dot(
                np.dot(transform.coeff_beta.T, self._matrices[1]), transform.coeff_beta
            )
        return OneBodyElectronicIntegrals(transform.final_basis, (matrix_a, matrix_b))

    def to_spin(self) -> np.ndarray:
        """Transforms the integrals into the special ``ElectronicBasis.SO`` basis.

        In this case of the 1-body integrals, the returned matrix is a block matrix of the form:
        ``[[alpha_spin, zeros], [zeros, beta_spin]]``.

        Returns:
            A single matrix containing the ``n-body`` integrals in the spin orbital basis.
        """
        if self._basis == ElectronicBasis.SO:
            return self._matrices  # type: ignore

        matrix_a = self._matrices[0]
        matrix_b = matrix_a if self._matrices[1] is None else self._matrices[1]
        zeros = np.zeros(matrix_a.shape)
        so_matrix = np.block([[matrix_a, zeros], [zeros, matrix_b]])

        return np.where(np.abs(so_matrix) > 1e-12, so_matrix, 0.0)

    def _calc_coeffs_with_ops(self, indices: Tuple[int, ...]) -> List[Tuple[int, str]]:
        return [(indices[0], "+"), (indices[1], "-")]
