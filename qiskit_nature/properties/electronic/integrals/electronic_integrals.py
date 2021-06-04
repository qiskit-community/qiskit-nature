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

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import itertools

import numpy as np

from qiskit_nature.operators.second_quantization import FermionicOp

from ..bases import ElectronicBasis


class ElectronicIntegrals(ABC):
    """TODO."""

    def __init__(
        self,
        num_body_terms: int,
        basis: ElectronicBasis,
        matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]],
    ) -> None:
        """TODO."""
        self._basis = basis
        assert num_body_terms >= 1
        self._num_body_terms = num_body_terms
        self._matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]]
        if basis == ElectronicBasis.SO:
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
    def _calc_coeffs_with_ops(self, idx) -> List[Tuple[complex, str]]:
        """TODO."""
        raise NotImplementedError("TODO.")

    def _create_base_ops(self) -> List[Tuple[str, complex]]:
        """TODO."""
        return self._create_base_ops_labels(
            self.to_spin(), 2 * self._num_body_terms, self._calc_coeffs_with_ops
        )

    def to_second_q_op(self) -> FermionicOp:
        """TODO."""
        base_ops_labels = self._create_base_ops()

        # TODO: allow an empty list as argument to FermionicOp
        fac = 2 if self._basis != ElectronicBasis.SO else 1
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
            base_op = ElectronicIntegrals._create_base_op_from_labels(
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
