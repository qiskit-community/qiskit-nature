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

""" Calculator of 1-body integrals for a total magnetization."""
from typing import Tuple

import numpy as np


def calc_total_magnetization_ints(num_modes: int) -> Tuple[np.ndarray, None]:
    """
    Calculates 1-body integrals for a total magnetization.

    Args:
        num_modes (int): Number of modes.

    Returns:
        Tuple(numpy.ndarray, None): 1-body integrals and None 2-body integrals for a total
        magnetization.
    """
    modes = num_modes
    h_1 = np.eye(modes, dtype=complex) * 0.5
    h_1[modes // 2 :, modes // 2 :] *= -1.0

    return h_1, None
