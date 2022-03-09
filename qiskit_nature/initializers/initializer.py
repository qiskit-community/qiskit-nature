# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Base Initializer class """

import abc
from typing import Sequence, Tuple

import numpy as np


class Initializer(abc.ABC):
    """Base Initializer class."""

    def __init__(self) -> None:
        super().__init__()
        self._coefficients = None
        self._energy_corrections = None

    @property
    def coefficients(self) -> np.ndarray:
        """Get the coefficients for the molecule.

        Returns:
            The initializer coefficients.
        """
        return self._coefficients

    def compute_corrections(
        self,
        excitations: Sequence,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the coefficients for the molecule.

        Returns:
            The initializer coefficients.
        """
        coeffs = np.zeroes(len(excitations))
        self._coefficients = coeffs
        self._energy_corrections = coeffs
        return coeffs, coeffs
