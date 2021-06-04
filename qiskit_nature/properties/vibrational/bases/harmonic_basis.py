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

import numpy as np

from .vibrational_basis import VibrationalBasis


class HarmonicBasis(VibrationalBasis):
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
