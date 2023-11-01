# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Drivers (:mod:`qiskit_nature.second_q.drivers`)
=============================================================

.. currentmodule:: qiskit_nature.second_q.drivers

Driver Base Class
=================
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseDriver
   VibrationalStructureDriver
   ElectronicStructureDriver

Driver Common
=============
.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MethodType
   InitialGuess

Drivers
=======

The drivers in the chemistry module obtain their information from classical ab-initio programs
or libraries. Several drivers, interfacing to common programs and libraries, are
available. To use the driver its dependent program/library must be installed. See
the relevant installation instructions below for your program/library that you intend
to use.

.. toctree::
   :maxdepth: 1

   qiskit_nature.second_q.drivers.gaussiand
   qiskit_nature.second_q.drivers.psi4d
   qiskit_nature.second_q.drivers.pyscfd

Electronic Structure Drivers
============================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GaussianDriver
   Psi4Driver
   PySCFDriver

Vibrational Structure Drivers
=============================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GaussianForcesDriver

General Driver
==============

The :class:`GaussianLogDriver` allows an arbitrary Gaussian Job Control File to be run and
return a :class:`GaussianLogResult` containing the log as well as ready access certain data
of interest that is parsed from the log.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GaussianLogDriver
   GaussianLogResult

"""

from .base_driver import BaseDriver
from .vibrational_structure_driver import VibrationalStructureDriver
from .electronic_structure_driver import ElectronicStructureDriver, MethodType
from .gaussiand import GaussianDriver, GaussianLogDriver, GaussianLogResult, GaussianForcesDriver
from .psi4d import Psi4Driver
from .pyscfd import PySCFDriver, InitialGuess

__all__ = [
    "MethodType",
    "BaseDriver",
    "VibrationalStructureDriver",
    "ElectronicStructureDriver",
    "GaussianDriver",
    "GaussianForcesDriver",
    "GaussianLogDriver",
    "GaussianLogResult",
    "Psi4Driver",
    "PySCFDriver",
    "InitialGuess",
]
