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

""" Calculator of 1- and 2-body integrals for a total angular momentum."""

import itertools

import numpy as np


def calc_total_ang_momentum_ints(num_modes):
    """
    Calculates 1- and 2-body integrals for a total angular momentum.

    Args:
        num_modes (int): Number of modes.

    Returns:
        Tuple(numpy.ndarray, numpy.ndarray): Tuple of 1- and 2-body integrals for a total angular
        momentum.
    """
    x_h1, x_h2 = _calc_s_x_squared_ints(num_modes)
    y_h1, y_h2 = _calc_s_y_squared_ints(num_modes)
    z_h1, z_h2 = _calc_s_z_squared_ints(num_modes)
    h_1 = x_h1 + y_h1 + z_h1
    h_2 = x_h2 + y_h2 + z_h2

    return h_1, h_2


# TODO eliminate code duplication below

def _calc_s_x_squared_ints(num_modes):
    num_modes_2 = num_modes // 2
    h_1 = np.zeros((num_modes, num_modes))
    h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

    for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
        if p != q:
            h_2[p, p + num_modes_2, q, q + num_modes_2] += 1.0
            h_2[p + num_modes_2, p, q, q + num_modes_2] += 1.0
            h_2[p, p + num_modes_2, q + num_modes_2, q] += 1.0
            h_2[p + num_modes_2, p, q + num_modes_2, q] += 1.0
        else:
            h_2[p, p + num_modes_2, p, p + num_modes_2] -= 1.0
            h_2[p + num_modes_2, p, p + num_modes_2, p] -= 1.0
            h_2[p, p, p + num_modes_2, p + num_modes_2] -= 1.0
            h_2[p + num_modes_2, p + num_modes_2, p, p] -= 1.0

            h_1[p, p] += 1.0
            h_1[p + num_modes_2, p + num_modes_2] += 1.0

    h_1 *= 0.25
    h_2 *= 0.25
    return h_1, h_2


def _calc_s_y_squared_ints(num_modes):
    num_modes_2 = num_modes // 2
    h_1 = np.zeros((num_modes, num_modes))
    h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

    for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
        if p != q:
            h_2[p, p + num_modes_2, q, q + num_modes_2] -= 1.0
            h_2[p + num_modes_2, p, q, q + num_modes_2] += 1.0
            h_2[p, p + num_modes_2, q + num_modes_2, q] += 1.0
            h_2[p + num_modes_2, p, q + num_modes_2, q] -= 1.0
        else:
            h_2[p, p + num_modes_2, p, p + num_modes_2] += 1.0
            h_2[p + num_modes_2, p, p + num_modes_2, p] += 1.0
            h_2[p, p, p + num_modes_2, p + num_modes_2] -= 1.0
            h_2[p + num_modes_2, p + num_modes_2, p, p] -= 1.0

            h_1[p, p] += 1.0
            h_1[p + num_modes_2, p + num_modes_2] += 1.0

    h_1 *= 0.25
    h_2 *= 0.25
    return h_1, h_2


def _calc_s_z_squared_ints(num_modes):
    num_modes_2 = num_modes // 2
    h_1 = np.zeros((num_modes, num_modes))
    h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

    for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
        if p != q:
            h_2[p, p, q, q] += 1.0
            h_2[p + num_modes_2, p + num_modes_2, q, q] -= 1.0
            h_2[p, p, q + num_modes_2, q + num_modes_2] -= 1.0
            h_2[p + num_modes_2, p + num_modes_2,
                q + num_modes_2, q + num_modes_2] += 1.0
        else:
            h_2[p, p + num_modes_2, p + num_modes_2, p] += 1.0
            h_2[p + num_modes_2, p, p, p + num_modes_2] += 1.0

            h_1[p, p] += 1.0
            h_1[p + num_modes_2, p + num_modes_2] += 1.0

    h_1 *= 0.25
    h_2 *= 0.25
    return h_1, h_2
