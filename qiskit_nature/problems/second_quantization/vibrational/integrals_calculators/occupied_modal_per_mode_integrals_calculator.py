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

""" Calculator of WatsonHamiltonian integrals for the number of occupied modals per mode."""

from typing import List, Tuple


def calc_occ_modals_per_mode_ints(
    num_modals: List[int], mode_index: int
) -> List[List[Tuple[List[List[int]], complex]]]:
    """
    Calculates WatsonHamiltonian-like integrals to evaluate the number of occupied modals in a given
    mode.

    Args:
        num_modals: the number of modals per mode.
        mode_index: the mode index.

    Returns:
        Tuple(numpy.ndarray, None): 1-body integrals and None 2-body integrals for a total
        particle number.
    """
    h_mat: List[List[Tuple[List[List[int]], complex]]] = [[]]
    for modal in range(num_modals[mode_index]):
        h_mat[0].append(([[mode_index, modal, modal]], 1.0))

    return h_mat
