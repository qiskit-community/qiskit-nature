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
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import itertools

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization import FermionicOp


class Basis(Enum):
    """TODO."""

    # pylint: disable=invalid-name
    AO = "ao"
    MO = "mo"
    SO = "so"


class BasisTransform:
    """TODO."""

    def __init__(
        self,
        initial_basis: Basis,
        final_basis: Basis,
        coeff_alpha: np.ndarray,
        coeff_beta: Optional[np.ndarray] = None,
    ) -> None:
        """TODO."""
        self._initial_basis = initial_basis
        self._final_basis = final_basis
        self._coeff_alpha = coeff_alpha
        self._coeff_beta = coeff_alpha if coeff_beta is None else coeff_beta


class _ElectronicIntegrals(ABC):
    """TODO."""

    def __init__(
        self,
        num_body_terms: int,
        basis: Basis,
        matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]],
    ) -> None:
        """TODO."""
        self._basis = basis
        assert num_body_terms >= 1
        self._num_body_terms = num_body_terms
        self._matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]]
        if basis == Basis.SO:
            assert isinstance(matrices, np.ndarray)
            self._matrices = matrices
        else:
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

        # TODO: allow an empty list as argument to FermionicOp
        fac = 2 if self._basis != Basis.SO else 1
        initial_label_with_ceoff = ("I" * fac * len(self._matrices[0]), 0)
        base_ops_labels.append(initial_label_with_ceoff)

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
    def _create_base_op_from_labels(coeff: complex, length: int, coeffs_with_ops) -> FermionicOp:
        label = ["I"] * length
        base_op = FermionicOp("".join(label)) * coeff
        for i, op in coeffs_with_ops:
            label_i = label.copy()
            label_i[i] = op
            base_op @= FermionicOp("".join(label_i))
        return base_op


class _1BodyElectronicIntegrals(_ElectronicIntegrals):
    """TODO."""

    def __init__(
        self, basis: Basis, matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]]
    ) -> None:
        """TODO."""
        super().__init__(1, basis, matrices)

    def transform_basis(self, transform: BasisTransform) -> _1BodyElectronicIntegrals:
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
        return _1BodyElectronicIntegrals(transform._final_basis, (matrix_a, matrix_b))

    def to_spin(self) -> np.ndarray:
        """TODO."""
        if self._basis == Basis.SO:
            return self._matrices  # type: ignore

        matrix_a = self._matrices[0]
        matrix_b = matrix_a if self._matrices[1] is None else self._matrices[1]
        zeros = np.zeros(matrix_a.shape)
        so_matrix = np.block([[matrix_a, zeros], [zeros, matrix_b]])

        return np.where(np.abs(so_matrix) > 1e-12, so_matrix, 0.0)

    def _create_base_ops(self) -> List[Tuple[str, complex]]:
        return self._create_base_ops_labels(self.to_spin(), 2, self._calc_coeffs_with_ops)

    def _calc_coeffs_with_ops(self, idx) -> List[Tuple[complex, str]]:
        return [(idx[0], "+"), (idx[1], "-")]


class _2BodyElectronicIntegrals(_ElectronicIntegrals):
    """TODO."""

    EINSUM_AO_TO_MO = "pqrs,pi,qj,rk,sl->ijkl"
    EINSUM_CHEM_TO_PHYS = "ijkl->ljik"

    def __init__(
        self, basis: Basis, matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]]
    ) -> None:
        """TODO."""
        super().__init__(2, basis, matrices)

    def transform_basis(self, transform: BasisTransform) -> _2BodyElectronicIntegrals:
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

        return _2BodyElectronicIntegrals(transform._final_basis, tuple(matrices))

    def to_spin(self) -> np.ndarray:
        """TODO."""
        if self._basis == Basis.SO:
            return self._matrices  # type: ignore

        so_matrix = np.zeros([2 * s for s in self._matrices[0].shape])
        one_indices = (
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (1, 1, 1, 1),
            (1, 0, 0, 1),
        )
        for ao_mat, one_idx in zip(self._matrices, one_indices):
            if ao_mat is None:
                ao_mat = self._matrices[0]
            phys_matrix = np.einsum(self.EINSUM_CHEM_TO_PHYS, ao_mat)
            kron = np.zeros((2, 2, 2, 2))
            kron[one_idx] = 1
            so_matrix -= 0.5 * np.kron(kron, phys_matrix)

        return np.where(np.abs(so_matrix) > 1e-12, so_matrix, 0.0)

    def _create_base_ops(self) -> List[Tuple[str, complex]]:
        return self._create_base_ops_labels(self.to_spin(), 4, self._calc_coeffs_with_ops)

    def _calc_coeffs_with_ops(self, idx) -> List[Tuple[complex, str]]:
        return [(idx[0], "+"), (idx[2], "+"), (idx[3], "-"), (idx[1], "-")]
