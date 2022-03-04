# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Initializer class """

import abc
import numpy as np
from typing import Sequence


class Initializer(abc.ABC):
    @property
    def coefficients(self) -> np.ndarray:
        """Get the coefficients for the molecule.

        Returns:
            The initializer coefficients.
        """
        return self._coefficients

    def compute_coefficients(
        self,
        excitations: Sequence,
    ) -> np.ndarray:
        coeffs = np.zeroes(len(excitations))
        self._coefficients = coeffs
        return coeffs
