# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Chemistry Core (:mod:`qiskit_nature.core`)
=============================================

.. currentmodule:: qiskit_nature.core

**DEPRECATED** See :mod:`qiskit_nature.transformations` which replace this.

The core was designed to be an extensible system that
took a :class:`~qiskit_nature.drivers.QMolecule`
and created output which was ready to be input directly to an Aqua algorithm
in the form of a qubit operator and list of auxiliary operators such as
dipole moments, spin, number of particles etc.

The one implementation here, :class:`Hamiltonian`, in essence wraps the
:class:`~qiskit_nature.FermionicOperator` to provide easier, convenient access to common
capabilities such that the :class:`~qiskit_nature.FermionicOperator` class need not be
used directly.

Core Base Class
===============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ChemistryOperator
   MolecularChemistryResult
   MolecularGroundStateResult
   MolecularExcitedStatesResult

Core
====

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Hamiltonian
   TransformationType
   QubitMappingType

"""

from .chemistry_operator import (ChemistryOperator, MolecularChemistryResult,
                                 MolecularGroundStateResult, MolecularExcitedStatesResult)
from .hamiltonian import Hamiltonian, TransformationType, QubitMappingType

__all__ = ['ChemistryOperator',
           'MolecularChemistryResult',
           'MolecularGroundStateResult',
           'MolecularExcitedStatesResult',
           'Hamiltonian',
           'TransformationType',
           'QubitMappingType']
