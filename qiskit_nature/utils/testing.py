# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Testing utilities."""

import numpy as np


def random_antisymmetric_matrix(dim: int):
    """Return a random antisymmetric matrix.

    Args:
        dim: The width and height of the matrix.
    """
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    return mat - mat.T
