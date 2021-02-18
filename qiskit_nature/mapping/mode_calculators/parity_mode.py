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
import numpy as np
from qiskit.quantum_info import Pauli


def parity_mode(num_modes: int):
    """
    Parity mode.

    Args:
        num_modes (int): number of modes

    Returns:
        list[Tuple]: Pauli
    """
    a_list = []
    for i in range(num_modes):
        a_z = [0] * (i - 1) + [1] if i > 0 else []
        a_x = [0] * (i - 1) + [0] if i > 0 else []
        b_z = [0] * (i - 1) + [0] if i > 0 else []
        b_x = [0] * (i - 1) + [0] if i > 0 else []
        a_z = np.asarray(a_z + [0] + [0] * (num_modes - i - 1), dtype=bool)
        a_x = np.asarray(a_x + [1] + [1] * (num_modes - i - 1), dtype=bool)
        b_z = np.asarray(b_z + [1] + [0] * (num_modes - i - 1), dtype=bool)
        b_x = np.asarray(b_x + [1] + [1] * (num_modes - i - 1), dtype=bool)
        a_list.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
    return a_list
