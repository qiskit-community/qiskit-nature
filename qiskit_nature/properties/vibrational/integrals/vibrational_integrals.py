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

from abc import ABC
from collections import Counter
from itertools import chain, cycle, permutations, product, tee
from typing import Dict, List, Tuple

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization import VibrationalOp

from ..bases import VibrationalBasis


class VibrationalIntegrals(ABC):
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
        self._basis: VibrationalBasis = None

    @property
    def basis(self) -> VibrationalBasis:
        """Returns the basis."""
        return self._basis

    @basis.setter
    def basis(self, basis: VibrationalBasis) -> None:
        """Sets the basis."""
        self._basis = basis

    def to_basis(self) -> np.ndarray:
        """TODO."""
        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)
        max_num_modals = max(num_modals_per_mode)

        matrix = np.zeros((num_modes, max_num_modals, max_num_modals) * self._num_body_terms)

        # we can cache already evaluated integrals to improve cases in which a basis is very
        # expensive to compute
        coeff_cache: Dict[Tuple[int, int, int, int, bool], float] = {}

        for coeff0, indices in self._integrals:
            assert len(set(indices)) == self._num_body_terms

            # NOTE: negative indices may be treated specially by a basis
            kinetic_term = any(index < 0 for index in indices)
            if kinetic_term:
                # once we have determined whether a term is kinetic, all indices must be positive
                indices = np.absolute(indices)

            # the number of times which an index occurs corresponds to the power of the operator
            powers = Counter(indices)

            index_list = []

            # we do an initial loop to evaluate all relevant basis integrals
            for mode, power in powers.items():
                iter_1, iter_2 = tee(zip(*np.tril_indices(num_modals_per_mode[mode - 1])))
                # we must store the indices of the mode in combination with all possible modal
                # permutations (lower triangular indices) for the next step
                index_list.append(zip(cycle([mode]), iter_1))
                for m, n in iter_2:
                    if (mode - 1, m, n, power, kinetic_term) in coeff_cache.keys():
                        # value already in cache
                        continue
                    coeff_cache[(mode - 1, m, n, power, kinetic_term)] = self.basis._eval_integral(
                        mode - 1, m, n, power, kinetic_term=kinetic_term
                    )

            # now we can iterate the product of all index lists (the cartesian product is equivalent
            # to nested for loops but has the benefit of being agnostic w.r.t. the number of body
            # terms)
            for index in product(*index_list):
                index_permutations = []
                coeff = coeff0
                for mode, (m, n) in index:
                    # compute the total coefficient
                    coeff *= coeff_cache[(mode - 1, m, n, powers[mode], kinetic_term)]
                    index_set = set()
                    # generate potentially symmetric permutations of the modal indices
                    for m_sub, n_sub in permutations((m, n)):
                        index_set.add((m_sub, n_sub))
                    index_permutations.append(
                        {(mode - 1, m_sub, n_sub) for (m_sub, n_sub) in index_set}
                    )

                if abs(coeff) > self.basis._threshold:
                    # update the matrix in all permuted locations
                    for i in product(*index_permutations):
                        matrix[tuple(chain(*i))] += coeff

        return matrix

    def to_second_q_op(self) -> VibrationalOp:
        """TODO."""
        if self._basis is None:
            raise QiskitNatureError("TODO")

        matrix = self.to_basis()
        labels = self._create_num_body_labels(matrix)

        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)

        if labels == []:
            # TODO: allow an empty list as argument to VibrationalOp
            initial_label_with_ceoff = ("I" * sum(num_modals_per_mode), 0)
            labels.append(initial_label_with_ceoff)

        return VibrationalOp(labels, num_modes, num_modals_per_mode)

    @staticmethod
    def _create_num_body_labels(matrix: np.ndarray) -> List[Tuple[str, complex]]:
        num_body_labels = []
        nonzero = np.nonzero(matrix)
        for coeff, indices in zip(matrix[nonzero], zip(*nonzero)):
            grouped_indices = sorted(
                [tuple(int(j) for j in indices[i : i + 3]) for i in range(0, len(indices), 3)]
            )
            coeff_label = VibrationalIntegrals._create_label_for_coeff(grouped_indices)
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
