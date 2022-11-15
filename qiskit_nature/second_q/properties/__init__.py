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
Properties (:mod:`qiskit_nature.second_q.properties`)
================================================================================================

.. currentmodule:: qiskit_nature.second_q.properties

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    SparseLabelOpsFactory
    Interpretable

Electronic Properties
---------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AngularMomentum
   ElectronicDensity
   ElectronicDipoleMoment
   Magnetization
   ParticleNumber

Vibrational Properties
----------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   OccupiedModals
"""

from .protocols import SparseLabelOpsFactory, Interpretable

from .angular_momentum import AngularMomentum
from .dipole_moment import ElectronicDipoleMoment
from .electronic_density import ElectronicDensity
from .magnetization import Magnetization
from .particle_number import ParticleNumber

from .occupied_modals import OccupiedModals


__all__ = [
    "SparseLabelOpsFactory",
    "Interpretable",
    "AngularMomentum",
    "ElectronicDensity",
    "ElectronicDipoleMoment",
    "Magnetization",
    "ParticleNumber",
    "OccupiedModals",
]
