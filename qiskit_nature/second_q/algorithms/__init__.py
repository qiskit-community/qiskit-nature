# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Algorithms (:mod:`qiskit_nature.second_q.algorithms`)
=====================================================

.. currentmodule:: qiskit_nature.second_q.algorithms

These are natural science algorithms to solve specific problems such as finding the ground state
energy, excited state energies or potential energy surfaces.

Excited State Solvers
+++++++++++++++++++++
Algorithms that can find the eigenvalues of an operator, e.g. excited states for chemistry.

The interface for such solvers,

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ExcitedStatesSolver

the solvers themselves

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ExcitedStatesEigensolver
   QEOM

and factories to provision quantum and/or classical algorithms upon which the above solvers may
depend

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   EigensolverFactory
   NumPyEigensolverFactory

Ground State Solvers
++++++++++++++++++++
Algorithms that can find the minimum eigenvalue of an operator, e.g. ground state for chemistry.

The interface for such solvers,

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GroundStateSolver

the solvers themselves

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GroundStateEigensolver

and factories to provision quantum and/or classical algorithms upon which the above solvers may
depend

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MinimumEigensolverFactory
   NumPyMinimumEigensolverFactory
   VQEUCCFactory
   VQEUVCCFactory

Initial Points
++++++++++++++
The factories linked above make use of utility classes to compute initial points to use with
specific ansatzes. More details may be found in the sub-module linked below.

.. autosummary::
   :toctree:

   initial_points
"""

from .excited_states_solvers import (
    ExcitedStatesEigensolver,
    ExcitedStatesSolver,
    QEOM,
    EigensolverFactory,
    NumPyEigensolverFactory,
)
from .ground_state_solvers import (
    GroundStateEigensolver,
    GroundStateSolver,
    MinimumEigensolverFactory,
    NumPyMinimumEigensolverFactory,
    VQEUCCFactory,
    VQEUVCCFactory,
)

__all__ = [
    "ExcitedStatesEigensolver",
    "ExcitedStatesSolver",
    "QEOM",
    "EigensolverFactory",
    "NumPyEigensolverFactory",
    "GroundStateEigensolver",
    "GroundStateSolver",
    "MinimumEigensolverFactory",
    "NumPyMinimumEigensolverFactory",
    "VQEUCCFactory",
    "VQEUVCCFactory",
]
