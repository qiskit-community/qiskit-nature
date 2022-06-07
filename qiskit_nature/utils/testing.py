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

from typing import Any

import numpy as np


def parse_random_seed(seed: Any) -> np.random.Generator:
    """Parse a random number generator seed and return a Generator.

    Args:
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`

    Returns:
        The np.random.Generator instance
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def random_antisymmetric_matrix(dim: int, seed: Any = None) -> np.ndarray:
    """Return a random antisymmetric matrix.

    Args:
        dim: The width and height of the matrix.

    Returns:
        The sampled antisymmetric matrix.
    """
    rng = parse_random_seed(seed)
    mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return mat - mat.T
