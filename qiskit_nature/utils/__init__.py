# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utilities (:mod:`qiskit_nature.utils`)
==============================================

.. currentmodule:: qiskit_nature.utils

Linear algebra utilities
------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   apply_matrix_to_slices
   givens_matrix
   double_factorized
   modified_cholesky
"""
import warnings

from .linalg import (
    apply_matrix_to_slices,
    double_factorized,
    givens_matrix,
    modified_cholesky,
)
from .opt_einsum import get_einsum

# Handles import for use of Qiskit parallel map to cater to location before
# and after the 1.0 version
try:
    # From 0.46 onwards parallel map is here
    from qiskit.utils import parallel_map as _parallel_map
except ImportError:
    # Until 0.46 it's here but in 0.46 raises a deprecation
    # using it at this older location, hence we try the new
    # location first above and only then fallback to the old here
    from qiskit.tools import parallel_map as _parallel_map

__all__ = [
    "apply_matrix_to_slices",
    "double_factorized",
    "givens_matrix",
    "get_einsum",
    "modified_cholesky",
]
