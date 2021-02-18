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


def jordan_wigner_mode(num_modes: int):
    r"""
    Jordan_Wigner mode.

    Each Fermionic Operator is mapped to 2 Pauli Operators, added together with the
    appropriate phase, i.e.:

    a_i\^\\dagger = Z\^i (X + iY) I\^(n-i-1) = (Z\^i X I\^(n-i-1)) + i (Z\^i Y I\^(n-i-1))
    a_i = Z\^i (X - iY) I\^(n-i-1)

    This is implemented by creating an array of tuples, each including two operators.
    The phase between two elements in a tuple is implicitly assumed, and added calculated at the
    appropriate time (see for example _one_body_mapping).

    Args:
        num_modes (int): number of modes

    Returns:
        list[Tuple]: Pauli
    """
    a_list = []
    for i in range(num_modes):
        a_z = np.asarray([1] * i + [0] + [0] * (num_modes - i - 1), dtype=bool)
        a_x = np.asarray([0] * i + [1] + [0] * (num_modes - i - 1), dtype=bool)
        b_z = np.asarray([1] * i + [1] + [0] * (num_modes - i - 1), dtype=bool)
        b_x = np.asarray([0] * i + [1] + [0] * (num_modes - i - 1), dtype=bool)
        a_list.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
    return a_list
