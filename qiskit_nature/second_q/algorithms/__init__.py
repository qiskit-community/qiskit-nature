# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2023.
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

and the specific raw result for the qEOM solver.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QEOMResult

The following factories are still available but have been **deprecated** in version 0.6.0 of Qiskit
Nature:

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

the solvers themselves.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GroundStateEigensolver

The following factories are still available but have been **deprecated** in version 0.6.0 of Qiskit
Nature:

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MinimumEigensolverFactory
   NumPyMinimumEigensolverFactory
   VQEUCCFactory
   VQEUVCCFactory

Initial Points
++++++++++++++
When using variational algorithms such as the :class:`~qiskit.algorithms.minimum_eigensolvers.VQE`
it may be necessary to set the initial parameters for the optimizer to a specific value (by default,
the optimizer will start from a random point). This depends on the problem one is trying to solve as
well as the ansatz used to solve the problem. To this extent, the following submodule provides
generator classes for such an ``initial_point``. For more information, refer to the documentation of
the submodule linked below as well as the how-to guides on
:ref:`using a UCC-like ansatz with a VQE <how-to-vqe-ucc>` or on
:ref:`using a UVCC-like ansatz with a VQE <how-to-vqe-uvcc>` for some specific examples.

.. autosummary::
   :toctree:

   initial_points
"""

from .excited_states_solvers import (
    ExcitedStatesEigensolver,
    ExcitedStatesSolver,
    QEOM,
    QEOMResult,
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
    "QEOMResult",
    "EigensolverFactory",
    "NumPyEigensolverFactory",
    "GroundStateEigensolver",
    "GroundStateSolver",
    "MinimumEigensolverFactory",
    "NumPyMinimumEigensolverFactory",
    "VQEUCCFactory",
    "VQEUVCCFactory",
]
