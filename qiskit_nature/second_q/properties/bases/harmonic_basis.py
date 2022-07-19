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

"""The Harmonic basis."""

from typing import Optional

import numpy as np

from .vibrational_basis import VibrationalBasis


class HarmonicBasis(VibrationalBasis):
    """The Harmonic basis.

    This class uses the Hermite polynomials (eigenstates of the harmonic oscillator) as a modal
    basis for the expression of the Watson Hamiltonian or any bosonic operator.

    References:

        [1] Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.
    """

    def eval_integral(
        self,
        mode: int,
        modal_1: int,
        modal_2: int,
        power: int,
        kinetic_term: bool = False,
    ) -> Optional[float]:
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
            ValueError: if the ``power`` exceeds 4.

        References:

            [1] J. Chem. Phys. 135, 134108 (2011)
                https://doi.org/10.1063/1.3644895 (Table 1)
        """
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

        coeff *= np.sqrt(2) ** power

        return None if abs(coeff) < self._threshold else coeff
