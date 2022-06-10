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
"""

from .linalg import apply_matrix_to_slices, givens_matrix

__all__ = [
    "apply_matrix_to_slices",
    "givens_matrix",
    "random_antisymmetric_matrix",
    "random_quadratic_hamiltonian",
]
