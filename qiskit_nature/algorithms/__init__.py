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
============================================
Algorithms (:mod:`qiskit_nature.algorithms`)
============================================

.. currentmodule:: qiskit_nature.algorithms

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

and factories to provision Quantum and/or Classical algorithms upon which the above solvers may
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

   AdaptVQE
   GroundStateEigensolver

and factories to provision Quantum and/or Classical algorithms upon which the above solvers may
depend

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MinimumEigensolverFactory
   NumPyMinimumEigensolverFactory
   VQEUCCFactory
   VQEUVCCFactory

Potential Energy Surface Samplers
+++++++++++++++++++++++++++++++++
Algorithms that can compute potential energy surfaces.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ~pes_samplers.BOPESSampler

The samplers include extrapolators to facilitate convergence across a set of points and support
of various potentials. More detail may be found in the sub-module linked below.

.. autosummary::
   :toctree:

   pes_samplers

Initial Points
++++++++++++++
The factories linked above make use of utility classes to compute initial points to use with
specific ansatzes. More detail may be found in the sub-module linked below.

.. autosummary::
   :toctree:

   initial_points

"""

from .pes_samplers import BOPESSampler

__all__ = [
    "BOPESSampler",
]
