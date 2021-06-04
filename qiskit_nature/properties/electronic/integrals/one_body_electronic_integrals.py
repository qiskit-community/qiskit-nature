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

"""TODO."""

from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError

from .electronic_integrals import ElectronicIntegrals
from ..bases import ElectronicBasis, ElectronicBasisTransform


class OneBodyElectronicIntegrals(ElectronicIntegrals):
    """TODO."""

    def __init__(
        self,
        basis: ElectronicBasis,
        matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]],
    ) -> None:
        """TODO."""
        super().__init__(1, basis, matrices)

    def transform_basis(self, transform: ElectronicBasisTransform) -> "OneBodyElectronicIntegrals":
        """TODO."""
        if self._basis == transform._final_basis:
            return self

        if self._basis != transform._initial_basis:
            raise QiskitNatureError("TODO")

        matrix_a = np.dot(
            np.dot(transform._coeff_alpha.T, self._matrices[0]), transform._coeff_alpha
        )
        matrix_b = None
        if self._matrices[1] is not None:
            matrix_b = np.dot(
                np.dot(transform._coeff_beta.T, self._matrices[1]), transform._coeff_beta
            )
        return OneBodyElectronicIntegrals(transform._final_basis, (matrix_a, matrix_b))

    def to_spin(self) -> np.ndarray:
        """TODO."""
        if self._basis == ElectronicBasis.SO:
            return self._matrices  # type: ignore

        matrix_a = self._matrices[0]
        matrix_b = matrix_a if self._matrices[1] is None else self._matrices[1]
        zeros = np.zeros(matrix_a.shape)
        so_matrix = np.block([[matrix_a, zeros], [zeros, matrix_b]])

        return np.where(np.abs(so_matrix) > 1e-12, so_matrix, 0.0)

    def _calc_coeffs_with_ops(self, idx) -> List[Tuple[complex, str]]:
        return [(idx[0], "+"), (idx[1], "-")]
