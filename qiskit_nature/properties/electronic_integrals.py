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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast, Dict, List, Optional, Tuple, Union

import itertools

import numpy as np

from qiskit_nature.operators.second_quantization import FermionicOp


class _ElectronicIntegrals(ABC):
    """TODO."""

    def __init__(
        self,
        num_body_terms: int,
        matrices: Tuple[Optional[np.ndarray], ...],
    ) -> None:
        """TODO."""
        assert num_body_terms >= 1
        self._num_body_terms = num_body_terms
        assert len(matrices) == 2 ** num_body_terms
        assert matrices[0] is not None
        self._matrices = matrices

    @abstractmethod
    def to_spin(self) -> np.ndarray:
        """TODO."""
        raise NotImplementedError("TODO.")

    @abstractmethod
    def _create_base_ops(self) -> List[Tuple[str, complex]]:
        """TODO."""
        raise NotImplementedError("TODO.")

    def to_second_q_op(self) -> FermionicOp:
        """TODO."""
        base_ops_labels = self._create_base_ops()
        return FermionicOp(base_ops_labels)

    @staticmethod
    def _create_base_ops_labels(
        integrals: np.ndarray, repeat_num: int, calc_coeffs_with_ops
    ) -> List[Tuple[str, complex]]:
        all_base_ops_labels = []
        integrals_length = len(integrals)
        for idx in itertools.product(range(integrals_length), repeat=repeat_num):
            coeff = integrals[idx]
            if not coeff:
                continue
            coeffs_with_ops = calc_coeffs_with_ops(idx)
            base_op = _ElectronicIntegrals._create_base_op_from_labels(
                coeff, integrals_length, coeffs_with_ops
            )
            all_base_ops_labels += base_op.to_list()
        return all_base_ops_labels

    @staticmethod
    def _create_base_op_from_labels(coeff, length: int, coeffs_with_ops) -> FermionicOp:
        label = ["I"] * length
        base_op = coeff * FermionicOp("".join(label))
        for i, op in coeffs_with_ops:
            label_i = label.copy()
            label_i[i] = op
            base_op @= FermionicOp("".join(label_i))
        return base_op


class _1BodyElectronicIntegrals(_ElectronicIntegrals):
    """TODO."""

    def __init__(self, matrices: Tuple[Optional[np.ndarray], ...]) -> None:
        """TODO."""
        super().__init__(1, matrices)

    def to_spin(self) -> np.ndarray:
        """TODO."""
        matrix_a = self._matrices[0]
        matrix_b = self._matrices[1] or matrix_a
        zeros = np.zeros(matrix_a.shape)
        return np.block([[matrix_a, zeros], [zeros, matrix_b]])

    def _create_base_ops(self) -> List[Tuple[str, complex]]:
        return self._create_base_ops_labels(self.to_spin(), 2, self._calc_coeffs_with_ops)

    def _calc_coeffs_with_ops(self, idx) -> List[Tuple[complex, str]]:
        return [(idx[0], "+"), (idx[1], "-")]


class _2BodyElectronicIntegrals(_ElectronicIntegrals):
    """TODO."""

    EINSUM_AO_TO_MO = "pqrs,pi,qj,rk,sl->ijkl"
    EINSUM_CHEM_TO_PHYS = "ijkl->ljik"

    def __init__(self, matrices: Tuple[Optional[np.ndarray], ...]) -> None:
        """TODO."""
        super().__init__(2, matrices)

    def to_spin(self) -> np.ndarray:
        """TODO."""
        so_matrix = np.zeros([2 * s for s in self._matrices[0].shape])
        one_indices = (
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1),
        )
        for ao_mat, one_idx in zip(self._matrices, one_indices):
            if ao_mat is None:
                ao_mat = self._matrices[0]
            phys_matrix = np.einsum(_2BodyElectronicIntegrals.EINSUM_CHEM_TO_PHYS, ao_mat)
            kron = np.zeros((2, 2, 2, 2))
            kron[one_idx] = 1
            so_matrix -= 0.5 * np.kron(kron, phys_matrix)
        return so_matrix

    def _create_base_ops(self) -> List[Tuple[str, complex]]:
        return self._create_base_ops_labels(self.to_spin(), 4, self._calc_coeffs_with_ops)

    def _calc_coeffs_with_ops(self, idx) -> List[Tuple[complex, str]]:
        return [(idx[0], "+"), (idx[2], "+"), (idx[3], "-"), (idx[1], "-")]
