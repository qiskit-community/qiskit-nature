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

from typing import Optional

import numpy as np

from .electronic_basis import ElectronicBasis


class ElectronicBasisTransform:
    """TODO."""

    def __init__(
        self,
        initial_basis: ElectronicBasis,
        final_basis: ElectronicBasis,
        coeff_alpha: np.ndarray,
        coeff_beta: Optional[np.ndarray] = None,
    ) -> None:
        """TODO."""
        self._initial_basis = initial_basis
        self._final_basis = final_basis
        self._coeff_alpha = coeff_alpha
        self._coeff_beta = coeff_alpha if coeff_beta is None else coeff_beta
