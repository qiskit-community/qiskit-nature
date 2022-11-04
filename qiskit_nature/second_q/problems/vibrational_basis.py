# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Vibrational basis base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache
from itertools import chain, cycle, permutations, product
from typing import Generator

import numpy as np


class VibrationalBasis(ABC):
    """The Vibrational basis base class.

    This class defines the interface which any vibrational basis must implement. A basis must be
    applied to the vibrational integrals in order to map them into a second-quantization form.

    The following attributes can be set via the initializer but can also be read and updated once
    the ``VibrationalBasis`` object has been constructed.

    Attributes:
        num_modals (list[int]): the number of modals into which each mode gets expanded in
            second-quantization.
        threshold (float): the threshold below which integral values will be dropped.
    """

    def __init__(
        self,
        num_modals: list[int],
        *,
        threshold: float = 1e-6,
    ) -> None:
        """
        Args:
            num_modals: the number of modals to be used for each mode.
            threshold: the threshold value below which an integral coefficient gets neglected.
        """
        self.num_modals = num_modals
        self.threshold = threshold

    @abstractmethod
    @lru_cache(maxsize=128)
    def eval_integral(
        self,
        mode: int,
        modal_1: int,
        modal_2: int,
        power: int,
        kinetic_term: bool = False,
    ) -> complex | None:
        """The integral evaluation method of this basis.

        Args:
            mode: the index of the mode.
            modal_1: the index of the first modal.
            modal_2: the index of the second modal.
            power: the exponent of the coordinate.
            kinetic_term: if this is True, the method should compute the integral of the kinetic
                term of the vibrational Hamiltonian, :math:``d^2/dQ^2``.

        Returns:
            The evaluated integral for the specified coordinate or ``None`` if this integral value
            falls below the threshold.

        Raises:
            ValueError: if an unsupported parameter is supplied.
        """

    def map(
        self, coefficient: complex, modes: tuple[int, ...]
    ) -> Generator[tuple[complex, tuple[int, ...]], None, None]:
        """Maps the provided coefficient and mode index to this second-quantization basis.

        This applies the actual basis and expands each mode into the number of modals with which the
        basis instance was initialized.

        Args:
            coefficient: the initial coefficient associated with the mode indices.
            modes: the mode indices. If all of these are negative, the coefficient is treated as
                belonging to a kinetic term.

        Yields:
            Pairs of integral values and indices. The indices are now three times as long as the
            initially provided modes index. The reason for that is that each mode index gets
            expanded into three indices, denoting the ``(mode, modal_1, modal_2)`` indices.
        """
        # negative indices may be treated specially by a basis
        kinetic_term = any(i < 0 for i in modes)

        # the number of times which an index occurs corresponds to the power of the operator
        powers = Counter(abs(i) for i in modes)

        # we generate the list of all possible modal permutations (lower triangular indices) for all
        # involved modes: each entry in this list is a list of tuples of the form:
        #   (mode_index, modal_index_1, modal_index_2)
        index_list = list(
            zip(cycle([mode]), *np.tril_indices(self.num_modals[mode - 1])) for mode in powers
        )

        # now we can iterate the product of all index lists (the cartesian product is equivalent to
        # nested for loops but has the benefit of being agnostic w.r.t. the number of body terms)
        for index in product(*index_list):
            index_permutations = []
            coeff = coefficient
            for mode, m, n in index:
                integral = self.eval_integral(
                    mode - 1, m, n, powers[mode], kinetic_term=kinetic_term
                )
                if integral is None:
                    break
                coeff *= integral
                # generate potentially symmetric permutations of the modal indices
                index_permutations.append(
                    {(mode - 1, m_sub, n_sub) for (m_sub, n_sub) in permutations((m, n))}
                )
            else:
                # update the matrix in all permuted locations
                for i in product(*index_permutations):
                    yield (coeff, tuple(chain(*i)))
