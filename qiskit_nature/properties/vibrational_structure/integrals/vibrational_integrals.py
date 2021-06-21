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

"""A container for arbitrary ``n-body`` vibrational integrals."""

from abc import ABC
from collections import Counter
from itertools import chain, cycle, permutations, product, tee
from typing import Dict, List, Optional, Tuple

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization import VibrationalOp

from ..bases import VibrationalBasis


class VibrationalIntegrals(ABC):
    """A container for arbitrary ``n-body`` vibrational integrals."""

    def __init__(
        self,
        num_body_terms: int,
        integrals: List[Tuple[float, Tuple[int, ...]]],
    ) -> None:
        """
        Args:
            num_body_terms: ``n``, as in the ``n-body`` terms stored in these integrals.
            integrals: a sparse list of integrals. The data format corresponds a list of pairs, with
                its first entry being the integral coefficient and the second entry being a tuple of
                integers of length ``num_body_terms``. These integers are the indices of the modes
                associated with the integral. If the indices are negative, the integral is treated
                as a kinetic term of the vibrational hamiltonian.

        Raises:
            ValueError if the number of body terms is less than 1.
        """
        if num_body_terms < 1:
            raise ValueError(
                "The number of body terms must be greater than 0, not '%s'.", num_body_terms
            )
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

    @property
    def integrals(self) -> List[Tuple[float, Tuple[int, ...]]]:
        """Returns the integrals."""
        return self._integrals

    @integrals.setter
    def integrals(self, integrals: List[Tuple[float, Tuple[int, ...]]]) -> None:
        """Sets the integrals."""
        self._integrals = integrals

    def to_basis(self) -> np.ndarray:
        """Maps the integrals into a basis which permits mapping into second-quantization.

        Returns:
            A single matrix containing the ``n-body`` integrals in the mapped basis.

        Raises:
            QiskitNatureError: if no basis has been set yet.
            ValueError: if a mis-matching integral set and number of body terms is encountered.
        """
        if self._basis is None:
            raise QiskitNatureError("You must set a basis first!")

        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)
        max_num_modals = max(num_modals_per_mode)

        matrix = np.zeros((num_modes, max_num_modals, max_num_modals) * self._num_body_terms)

        # we can cache already evaluated integrals to improve cases in which a basis is very
        # expensive to compute
        coeff_cache: Dict[Tuple[int, int, int, int, bool], Optional[float]] = {}

        for coeff0, indices in self._integrals:
            if len(set(indices)) != self._num_body_terms:
                raise ValueError(
                    "The number of body terms, %s, does not match the number of different indices "
                    "in your integral, %s.",
                    self._num_body_terms,
                    len(set(indices)),
                )

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
                    coeff_cache[(mode - 1, m, n, power, kinetic_term)] = self.basis.eval_integral(
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
                    cached_coeff = coeff_cache[(mode - 1, m, n, powers[mode], kinetic_term)]
                    if cached_coeff is None:
                        break
                    coeff *= cached_coeff
                    index_set = set()
                    # generate potentially symmetric permutations of the modal indices
                    for m_sub, n_sub in permutations((m, n)):
                        index_set.add((m_sub, n_sub))
                    index_permutations.append(
                        {(mode - 1, m_sub, n_sub) for (m_sub, n_sub) in index_set}
                    )
                else:
                    # update the matrix in all permuted locations
                    for i in product(*index_permutations):
                        matrix[tuple(chain(*i))] += coeff

        return matrix

    def to_second_q_op(self) -> VibrationalOp:
        """Creates the operator representing the Hamiltonian defined by these vibrational integrals.

        Returns:
            The ``VibrationalOp`` given by these vibrational integrals.

        Raises:
            QiskitNatureError: if no basis has been set yet.
        """
        try:
            matrix = self.to_basis()
        except QiskitNatureError as exc:
            raise QiskitNatureError() from exc

        num_modals_per_mode = self.basis._num_modals_per_mode
        num_modes = len(num_modals_per_mode)

        nonzero = np.nonzero(matrix)

        if not np.any(np.asarray(nonzero)):
            return VibrationalOp.zero(num_modes, num_modals_per_mode)

        labels = []

        for coeff, indices in zip(matrix[nonzero], zip(*nonzero)):
            # the indices need to be grouped into triplets of the form: (mode, modal_1, modal_2)
            grouped_indices = [
                tuple(int(j) for j in indices[i : i + 3]) for i in range(0, len(indices), 3)
            ]
            # the index groups need to processed in sorted order to produce a valid label
            coeff_label = self._create_label_for_coeff(sorted(grouped_indices))
            labels.append((coeff_label, coeff))

        return VibrationalOp(labels, num_modes, num_modals_per_mode)

    @staticmethod
    def _create_label_for_coeff(indices: List[Tuple[int, ...]]) -> str:
        """Generates the operator label for the given indices.

        Args:
            indices: A list of index triplets, where the first number is the mode index and the
                second and third numbers are the modal indices of that mode.

        Returns:
            The constructed operator label.
        """
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
