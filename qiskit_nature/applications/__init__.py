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
Chemistry Applications (:mod:`qiskit_nature.applications`)
=============================================================
These are chemistry applications leveraging quantum algorithms

.. currentmodule:: qiskit_nature.applications

Applications
============

**DEPRECATED** See Ground state solvers in :mod:`qiskit_nature.algorithms` which replace this.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MolecularGroundStateEnergy

"""

from .molecular_ground_state_energy import MolecularGroundStateEnergy

__all__ = [
    'MolecularGroundStateEnergy'
]
