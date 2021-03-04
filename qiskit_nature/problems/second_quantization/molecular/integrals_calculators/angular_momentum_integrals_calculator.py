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
from typing import Tuple, List

import numpy as np


def calc_total_ang_momentum_ints(num_modes: int) -> Tuple[np.ndarray, np.ndarray]:
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


def _calc_s_x_squared_ints(num_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    return _calc_squared_ints(num_modes, _modify_s_x_squared_ints_neq, _modify_s_x_squared_ints_eq)


def _calc_s_y_squared_ints(num_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    return _calc_squared_ints(num_modes, _modify_s_y_squared_ints_neq, _modify_s_y_squared_ints_eq)


def _calc_s_z_squared_ints(num_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    return _calc_squared_ints(num_modes, _modify_s_z_squared_ints_neq, _modify_s_z_squared_ints_eq)


def _calc_squared_ints(num_modes: int, func_neq, func_eq) -> Tuple[np.ndarray, np.ndarray]:
    # calculates 1- and 2-body integrals for a given angular momentum axis (x or y or z,
    # specified by func_neq and func_eq)
    num_modes_2 = num_modes // 2
    h_1 = np.zeros((num_modes, num_modes))
    h_2 = np.zeros((num_modes, num_modes, num_modes, num_modes))

    for p, q in itertools.product(range(num_modes_2), repeat=2):  # pylint: disable=invalid-name
        if p != q:
            h_2 = func_neq(h_2, p, q, num_modes_2)
        else:
            h_2 = func_eq(h_2, p, num_modes_2)
            h_1[p, p] += 1.0
            h_1[p + num_modes_2, p + num_modes_2] += 1.0
    h_1 *= 0.25
    h_2 *= 0.25
    return h_1, h_2


def _modify_s_x_squared_ints_neq(h_2: np.ndarray, p_ind: int, q_ind: int,
                                 num_modes_2: int) -> np.ndarray:
    indices = [(p_ind, p_ind + num_modes_2, q_ind, q_ind + num_modes_2),
               (p_ind + num_modes_2, p_ind, q_ind, q_ind + num_modes_2),
               (p_ind, p_ind + num_modes_2, q_ind + num_modes_2, q_ind),
               (p_ind + num_modes_2, p_ind, q_ind + num_modes_2, q_ind)]
    values = [1, 1, 1, 1]
    # adds provided values to values of 2-body integrals (x axis of angular momentum) at given
    # indices in case p neq q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_x_squared_ints_eq(h_2: np.ndarray, p_ind: int, num_modes_2: int) -> np.ndarray:
    indices = [(p_ind, p_ind + num_modes_2, p_ind, p_ind + num_modes_2),
               (p_ind + num_modes_2, p_ind, p_ind + num_modes_2, p_ind),
               (p_ind, p_ind, p_ind + num_modes_2, p_ind + num_modes_2),
               (p_ind + num_modes_2, p_ind + num_modes_2, p_ind, p_ind)]
    values = [-1, -1, -1, -1]
    # adds provided values to values of 2-body integrals (x axis of angular momentum) at given
    # indices in case p eq q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_y_squared_ints_neq(h_2: np.ndarray, p_ind: int, q_ind: int,
                                 num_modes_2: int) -> np.ndarray:
    indices = [(p_ind, p_ind + num_modes_2, q_ind, q_ind + num_modes_2),
               (p_ind + num_modes_2, p_ind, q_ind, q_ind + num_modes_2),
               (p_ind, p_ind + num_modes_2, q_ind + num_modes_2, q_ind),
               (p_ind + num_modes_2, p_ind, q_ind + num_modes_2, q_ind)]
    values = [-1, 1, 1, -1]
    # adds provided values to values of 2-body integrals (y axis of angular momentum) at given
    # indices in case p neq q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_y_squared_ints_eq(h_2: np.ndarray, p_ind: int, num_modes_2: int) -> np.ndarray:
    indices = [(p_ind, p_ind + num_modes_2, p_ind, p_ind + num_modes_2),
               (p_ind + num_modes_2, p_ind, p_ind + num_modes_2, p_ind),
               (p_ind, p_ind, p_ind + num_modes_2, p_ind + num_modes_2),
               (p_ind + num_modes_2, p_ind + num_modes_2, p_ind, p_ind)]
    values = [1, 1, -1, -1]
    # adds provided values to values of 2-body integrals (y axis of angular momentum) at given
    # indices in case p eq q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_z_squared_ints_neq(h_2: np.ndarray, p_ind: int, q_ind: int,
                                 num_modes_2: int) -> np.ndarray:
    indices = [(p_ind, p_ind, q_ind, q_ind),
               (p_ind + num_modes_2, p_ind + num_modes_2, q_ind, q_ind),
               (p_ind, p_ind, q_ind + num_modes_2, q_ind + num_modes_2),
               (p_ind + num_modes_2, p_ind + num_modes_2, q_ind + num_modes_2, q_ind + num_modes_2)]
    values = [1, -1, -1, 1]
    # adds provided values to values of 2-body integrals (z axis of angular momentum) at given
    # indices in case p neq q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _modify_s_z_squared_ints_eq(h_2: np.ndarray, p_ind: int, num_modes_2: int) -> np.ndarray:
    indices = [(p_ind, p_ind + num_modes_2, p_ind + num_modes_2, p_ind),
               (p_ind + num_modes_2, p_ind, p_ind, p_ind + num_modes_2)]
    values = [1, 1]
    # adds provided values to values of 2-body integrals (z axis of angular momentum) at given
    # indices in case p eq q
    return _add_values_to_s_squared_ints(h_2, indices, values)


def _add_values_to_s_squared_ints(h_2: np.ndarray, indices: List[Tuple[int, int, int, int]],
                                  values: List[int]) -> np.ndarray:
    for index, value in zip(indices, values):
        h_2[index] += value
    return h_2
