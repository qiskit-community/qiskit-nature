# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Ground State Solving Algorithms (:mod:`qiskit_nature.algorithms.ground_state_solvers`)
======================================================================================

.. currentmodule:: qiskit_nature.algorithms.ground_state_solvers

.. autosummary::
   :toctree: ../stubs/

   minimum_eigensolver_factories

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GroundStateEigensolver
   AdaptVQE
"""

from .ground_state_solver import GroundStateSolver
from .adapt_vqe import AdaptVQE
from .ground_state_eigensolver import GroundStateEigensolver
from .minimum_eigensolver_factories import (MinimumEigensolverFactory,
                                            NumPyMinimumEigensolverFactory,
                                            VQEUCCFactory,
                                            VQEUVCCFactory)

__all__ = ['GroundStateSolver',
           'AdaptVQE',
           'GroundStateEigensolver',
           'MinimumEigensolverFactory',
           'NumPyMinimumEigensolverFactory',
           'VQEUCCFactory',
           'VQEUVCCFactory',
           ]
