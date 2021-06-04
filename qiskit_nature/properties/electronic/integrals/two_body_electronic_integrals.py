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


class TwoBodyElectronicIntegrals(ElectronicIntegrals):
    """TODO."""

    EINSUM_AO_TO_MO = "pqrs,pi,qj,rk,sl->ijkl"
    EINSUM_CHEM_TO_PHYS = "ijkl->ljik"

    def __init__(
        self,
        basis: ElectronicBasis,
        matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]],
    ) -> None:
        """TODO."""
        super().__init__(2, basis, matrices)

    def transform_basis(
        self, transform: ElectronicBasisTransform
    ) -> "TwoBodyElectronicIntegrals":
        """TODO."""
        if self._basis == transform._final_basis:
            return self

        if self._basis != transform._initial_basis:
            raise QiskitNatureError("TODO")

        coeff_alpha = transform._coeff_alpha
        coeff_beta = transform._coeff_beta

        coeff_list = [
            (coeff_alpha, coeff_alpha, coeff_alpha, coeff_alpha),
            (coeff_alpha, coeff_alpha, coeff_beta, coeff_beta),
            (coeff_beta, coeff_beta, coeff_beta, coeff_beta),
            (coeff_beta, coeff_beta, coeff_alpha, coeff_alpha),
        ]
        matrices: List[Optional[np.ndarray]] = []
        for mat, coeffs in zip(self._matrices, coeff_list):
            if mat is None:
                matrices.append(None)
                continue
            matrices.append(np.einsum(self.EINSUM_AO_TO_MO, mat, *coeffs))

        return TwoBodyElectronicIntegrals(transform._final_basis, tuple(matrices))

    def to_spin(self) -> np.ndarray:
        """TODO."""
        if self._basis == ElectronicBasis.SO:
            return self._matrices  # type: ignore

        so_matrix = np.zeros([2 * s for s in self._matrices[0].shape])
        one_indices = (
            (0, 0, 0, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1),
            (0, 1, 1, 0),
        )
        for ao_mat, one_idx in zip(self._matrices, one_indices):
            if ao_mat is None:
                ao_mat = self._matrices[0]
            phys_matrix = np.einsum(self.EINSUM_CHEM_TO_PHYS, ao_mat)
            kron = np.zeros((2, 2, 2, 2))
            kron[one_idx] = 1
            so_matrix -= 0.5 * np.kron(kron, phys_matrix)

        return np.where(np.abs(so_matrix) > 1e-12, so_matrix, 0.0)

    def _calc_coeffs_with_ops(self, idx) -> List[Tuple[complex, str]]:
        return [(idx[0], "+"), (idx[2], "+"), (idx[3], "-"), (idx[1], "-")]
