# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
r"""
Vibrational Structure Properties (:mod:`qiskit_nature.properties.second_q.vibrational`)
==================================================================================================

.. currentmodule:: qiskit_nature.properties.second_q.vibrational

This module provides commonly evaluated properties for *vibrational* problems.
It also includes the default return object for the vibrational structure drivers:

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   VibrationalStructureDriverResult

The main :class:`~qiskit_nature.properties.Property` of this module is the

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   VibrationalEnergy

which constructs the primary Hamiltonian whose solution is the goal of the Quantum Algorithm.
In order to ensure a physical solution, the following auxiliary property is also evaluated by
default, through which the algorithm can ensure that the number of particles is being conserved:

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   OccupiedModals


Utilities
+++++++++

.. autosummary::
   :toctree:

   bases
   integrals

"""

from .occupied_modals import OccupiedModals
from .vibrational_structure_driver_result import VibrationalStructureDriverResult
from .vibrational_energy import VibrationalEnergy

__all__ = [
    "OccupiedModals",
    "VibrationalStructureDriverResult",
    "VibrationalEnergy",
]
