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
"""Loads the energy matrix from the Miyazawa-Jernigan potential file."""
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np


def _load_energy_matrix_file() -> Tuple[np.ndarray, List[str]]:
    """Returns the energy matrix from the Miyazawa-Jernigan potential file."""

    path = _construct_resource_path()
    matrix = np.loadtxt(fname=path, dtype=str)
    energy_matrix = _parse_energy_matrix(matrix)
    symbols = list(matrix[0, :])
    return energy_matrix, symbols


def _construct_resource_path():
    current_path_parent = Path(__file__).resolve().parent.parent
    resource_relative_path = Path("interactions/resources")
    current_path_parent = os.path.join(current_path_parent, resource_relative_path)
    filename = "mj_matrix.txt"
    path = os.path.normpath(os.path.join(current_path_parent, filename))
    return path


def _parse_energy_matrix(matrix: np.ndarray) -> np.ndarray:
    """Parses a matrix loaded from the Miyazawa-Jernigan potential file."""
    energy_matrix = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]))
    for row in range(1, np.shape(matrix)[0]):
        for col in range(row - 1, np.shape(matrix)[1]):
            energy_matrix[row, col] = float(matrix[row, col])
    energy_matrix = energy_matrix[
        1:,
    ]
    return energy_matrix
