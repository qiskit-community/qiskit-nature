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
from enum import Enum
from typing import cast, Dict, List, Optional, Tuple, Union

import itertools

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization import VibrationalOp


class BosonicBasis(ABC):
    """TODO."""

    def __init__(
        self,
        num_modals_per_mode: List[int],
        threshold: float = 1e-6,
    ) -> None:
        """TODO."""
        self._num_modals_per_mode = num_modals_per_mode
        self._threshold = threshold

    @abstractmethod
    def _eval_integral(
        self,
        mode: int,
        modal_1: int,
        modal_2: int,
        power: int,
        kinetic_term: bool = False,
    ) -> float:
        """TODO."""


class HarmonicBasis(BosonicBasis):
    """TODO."""

    def _eval_integral(
        self,
        mode: int,
        modal_1: int,
        modal_2: int,
        power: int,
        kinetic_term: bool = False,
    ) -> float:
        """TODO."""
        coeff = 0.0

        if power == 1:
            if modal_1 - modal_2 == 1:
                coeff = np.sqrt(modal_1 / 2)
        elif power == 2:
            if modal_1 - modal_2 == 0:
                coeff = (modal_1 + 1 / 2) * (-1.0 if kinetic_term else 1.0)
            elif modal_1 - modal_2 == 2:
                coeff = np.sqrt(modal_1 * (modal_1 - 1)) / 2
        elif power == 3:
            if modal_1 - modal_2 == 1:
                coeff = 3 * np.power(modal_1 / 2, 3 / 2)
            elif modal_1 - modal_2 == 3:
                coeff = np.sqrt(modal_1 * (modal_1 - 1) * (modal_1 - 2)) / np.power(2, 3 / 2)
        elif power == 4:
            if modal_1 - modal_2 == 0:
                coeff = (6 * modal_1 * (modal_1 + 1) + 3) / 4
            elif modal_1 - modal_2 == 2:
                coeff = (modal_1 - 1 / 2) * np.sqrt(modal_1 * (modal_1 - 1))
            elif modal_1 - modal_2 == 4:
                coeff = np.sqrt(modal_1 * (modal_1 - 1) * (modal_1 - 2) * (modal_1 - 3)) / 4
        else:
            raise ValueError("The Q power is to high, only up to 4 is currently supported.")

        return coeff * (np.sqrt(2) ** power)


class _VibrationalIntegrals(ABC):
    """TODO."""

    def __init__(
        self,
        num_body_terms: int,
        integrals: List[Tuple[float, Tuple[int, ...]]],
    ) -> None:
        """TODO."""
        assert num_body_terms >= 1
        self._num_body_terms = num_body_terms
        self._integrals = integrals
        self._basis: BosonicBasis = None

    @property
    def basis(self) -> BosonicBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: BosonicBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    @abstractmethod
    def to_basis(self) -> np.ndarray:
        """TODO."""
        raise NotImplementedError("TODO.")

    def to_second_q_op(self) -> VibrationalOp:
        """TODO."""
        if self._basis is None:
            raise QiskitNatureError("TODO")

        matrix = self.to_basis()
        labels = self._create_num_body_labels(matrix)

        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)
        return VibrationalOp(labels, num_modes, num_modals_per_mode)

    @staticmethod
    def _create_num_body_labels(matrix: np.ndarray) -> List[Tuple[str, complex]]:
        num_body_labels = []
        nonzero = np.nonzero(matrix)
        for coeff, indices in zip(matrix[nonzero], zip(*nonzero)):
            grouped_indices = sorted(
                [tuple(int(j) for j in indices[i : i + 3]) for i in range(0, len(indices), 3)]
            )
            coeff_label = _VibrationalIntegrals._create_label_for_coeff(grouped_indices)
            num_body_labels.append((coeff_label, coeff))
        return num_body_labels

    @staticmethod
    def _create_label_for_coeff(indices: List[Tuple[int, ...]]) -> str:
        complete_labels_list = []
        for mode, modal_raise, modal_lower in indices:
            if modal_raise <= modal_lower:
                complete_labels_list.append(f"+_{mode}*{modal_raise}")
                complete_labels_list.append(f"-_{mode}*{modal_lower}")
            else:
                complete_labels_list.append(f"-_{mode}*{modal_lower}")
                complete_labels_list.append(f"+_{mode}*{modal_raise}")
        complete_label = " ".join(complete_labels_list)
        return complete_label


class _1BodyVibrationalIntegrals(_VibrationalIntegrals):
    """TODO."""

    def to_basis(self) -> np.ndarray:
        """TODO."""
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)
        max_num_modals = max(num_modals_per_mode)

        matrix = np.zeros((num_modes, max_num_modals, max_num_modals))

        for coeff0, indices in self._integrals:  # Entry is coeff (float) followed by indices (ints)
            assert len(set(indices)) == 1
            index = indices[0]
            power = len(indices)

            # NOTE: negative indices may be treated specially by a basis
            kinetic_term = index < 0
            if kinetic_term:
                index = -index

            local_num_modals = num_modals_per_mode[index - 1]
            for m in range(local_num_modals):
                for n in range(m + 1):

                    coeff = coeff0 * self.basis._eval_integral(
                        index - 1, m, n, power, kinetic_term=kinetic_term
                    )

                    if abs(coeff) > self.basis._threshold:
                        matrix[index - 1, m, n] += coeff
                        if m != n:
                            matrix[index - 1, n, m] += coeff

        return matrix


class _2BodyVibrationalIntegrals(_VibrationalIntegrals):
    """TODO."""

    def to_basis(self) -> np.ndarray:
        """TODO."""
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)
        max_num_modals = max(num_modals_per_mode)

        matrix = np.zeros(
            (num_modes, max_num_modals, max_num_modals, num_modes, max_num_modals, max_num_modals)
        )

        for coeff0, indices in self._integrals:  # Entry is coeff (float) followed by indices (ints)
            assert len(set(indices)) == 2

            kinetic_term = False

            # Note: these negative indices as detected below are explicitly generated in
            # _compute_modes for other potential uses. They are not wanted by this logic.
            if any(index < 0 for index in indices):
                kinetic_term = True
                indices = np.absolute(indices)
            index_dict = {}  # type: Dict[int, int]
            for i in indices:
                if index_dict.get(i) is None:
                    index_dict[i] = 1
                else:
                    index_dict[i] += 1

            modes = list(index_dict.keys())

            for m in range(num_modals_per_mode[modes[0] - 1]):
                for n in range(m + 1):
                    coeff1 = coeff0 * self.basis._eval_integral(
                        modes[0] - 1, m, n, index_dict[modes[0]], kinetic_term=kinetic_term
                    )
                    for j in range(num_modals_per_mode[modes[1] - 1]):
                        for k in range(j + 1):
                            coeff = coeff1 * self.basis._eval_integral(
                                modes[1] - 1,
                                j,
                                k,
                                index_dict[modes[1]],
                                kinetic_term=kinetic_term,
                            )
                            if abs(coeff) > self.basis._threshold:
                                matrix[modes[0] - 1, m, n, modes[1] - 1, j, k] += coeff
                                if m != n:
                                    matrix[modes[0] - 1, n, m, modes[1] - 1, j, k] += coeff
                                if j != k:
                                    matrix[modes[0] - 1, m, n, modes[1] - 1, k, j] += coeff
                                if m != n and j != k:
                                    matrix[modes[0] - 1, n, m, modes[1] - 1, k, j] += coeff

        return matrix


class _3BodyVibrationalIntegrals(_VibrationalIntegrals):
    """TODO."""

    def to_basis(self) -> np.ndarray:
        """TODO."""
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)
        max_num_modals = max(num_modals_per_mode)

        matrix = np.zeros(
            (
                num_modes,
                max_num_modals,
                max_num_modals,
                num_modes,
                max_num_modals,
                max_num_modals,
                num_modes,
                max_num_modals,
                max_num_modals,
            )
        )

        for coeff0, indices in self._integrals:  # Entry is coeff (float) followed by indices (ints)
            assert len(set(indices)) == 3

            kinetic_term = False

            # Note: these negative indices as detected below are explicitly generated in
            # _compute_modes for other potential uses. They are not wanted by this logic.
            if any(index < 0 for index in indices):
                kinetic_term = True
                indices = np.absolute(indices)
            index_dict = {}  # type: Dict[int, int]
            for i in indices:
                if index_dict.get(i) is None:
                    index_dict[i] = 1
                else:
                    index_dict[i] += 1

            modes = list(index_dict.keys())

            for m in range(num_modals_per_mode[modes[0] - 1]):
                for n in range(m + 1):
                    coeff1 = coeff0 * self.basis._eval_integral(
                        modes[0] - 1, m, n, index_dict[modes[0]], kinetic_term=kinetic_term
                    )
                    for j in range(num_modals_per_mode[modes[1] - 1]):
                        for k in range(j + 1):
                            coeff2 = coeff1 * self.basis._eval_integral(
                                modes[1] - 1,
                                j,
                                k,
                                index_dict[modes[1]],
                                kinetic_term=kinetic_term,
                            )
                            # pylint: disable=locally-disabled, invalid-name
                            for p in range(num_modals_per_mode[modes[2] - 1]):
                                for q in range(p + 1):
                                    coeff = coeff2 * self.basis._eval_integral(
                                        modes[2] - 1,
                                        p,
                                        q,
                                        index_dict[modes[2]],
                                        kinetic_term=kinetic_term,
                                    )
                                    if abs(coeff) > self.basis._threshold:
                                        matrix[
                                            modes[0] - 1,
                                            m,
                                            n,
                                            modes[1] - 1,
                                            j,
                                            k,
                                            modes[2] - 1,
                                            p,
                                            q,
                                        ] += coeff
                                        if m != n:
                                            matrix[
                                                modes[0] - 1,
                                                n,
                                                m,
                                                modes[1] - 1,
                                                j,
                                                k,
                                                modes[2] - 1,
                                                p,
                                                q,
                                            ] += coeff
                                        if k != j:
                                            matrix[
                                                modes[0] - 1,
                                                m,
                                                n,
                                                modes[1] - 1,
                                                k,
                                                j,
                                                modes[2] - 1,
                                                p,
                                                q,
                                            ] += coeff
                                        if p != q:
                                            matrix[
                                                modes[0] - 1,
                                                m,
                                                n,
                                                modes[1] - 1,
                                                j,
                                                k,
                                                modes[2] - 1,
                                                q,
                                                p,
                                            ] += coeff
                                        if m != n and k != j:
                                            matrix[
                                                modes[0] - 1,
                                                n,
                                                m,
                                                modes[1] - 1,
                                                k,
                                                j,
                                                modes[2] - 1,
                                                p,
                                                q,
                                            ] += coeff
                                        if m != n and p != q:
                                            matrix[
                                                modes[0] - 1,
                                                n,
                                                m,
                                                modes[1] - 1,
                                                j,
                                                k,
                                                modes[2] - 1,
                                                q,
                                                p,
                                            ] += coeff
                                        if p != q and k != j:
                                            matrix[
                                                modes[0] - 1,
                                                m,
                                                n,
                                                modes[1] - 1,
                                                k,
                                                j,
                                                modes[2] - 1,
                                                q,
                                                p,
                                            ] += coeff
                                        if m != n and j != k and p != q:
                                            matrix[
                                                modes[0] - 1,
                                                n,
                                                m,
                                                modes[1] - 1,
                                                k,
                                                j,
                                                modes[2] - 1,
                                                q,
                                                p,
                                            ] += coeff

        return matrix
