# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
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

from .linalg import (
    apply_matrix_to_slices,
    double_factorized,
    givens_matrix,
    modified_cholesky,
)
from .opt_einsum import get_einsum

__all__ = [
    "apply_matrix_to_slices",
    "double_factorized",
    "givens_matrix",
    "get_einsum",
    "modified_cholesky",
]
