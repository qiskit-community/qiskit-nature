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
Testing utilities (:mod:`qiskit_nature.testing`)
================================================

.. currentmodule:: qiskit_nature.testing

Random sampling utilities
-------------------------

.. autosummary::

   random_antisymmetric_matrix
   random_quadratic_hamiltonian
"""

from .random import random_antisymmetric_matrix, random_quadratic_hamiltonian

__all__ = [
    "random_antisymmetric_matrix",
    "random_quadratic_hamiltonian",
]
