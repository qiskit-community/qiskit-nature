# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
r"""
Electronic Structure Properties (:mod:`qiskit_nature.properties.electronic_structure`)
======================================================================================

.. currentmodule:: qiskit_nature.properties.electronic_structure

This module provides commonly evaluated properties for *electronic* problems.

The main ``Property`` of this module is the

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ElectronicEnergy

which constructs the primary Hamiltonian whose solution is the goal of the Quantum Algorithm.
The following auxiliary properties will be evaluated by default to provide further details of the
solution:

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ParticleNumber
   AngularMomentum
   Magnetization

These properties are required if you want to ensure that the number of particles in your system is
being conserved.

Finally, if the driver which you used provided dipole moment integrals, this property will also be
evaluated:

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   TotalDipoleMoment


Utilities
+++++++++

.. autosummary::
   :toctree:

   bases
   integrals

"""

from .angular_momentum import AngularMomentum
from .dipole_moment import TotalDipoleMoment
from .electronic_energy import ElectronicEnergy
from .magnetization import Magnetization
from .particle_number import ParticleNumber

__all__ = [
    "AngularMomentum",
    "TotalDipoleMoment",
    "ElectronicEnergy",
    "Magnetization",
    "ParticleNumber",
]
