# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Second Quantization Drivers (:mod:`qiskit_nature.drivers.second_q`)
==============================================================================
.. currentmodule:: qiskit_nature.drivers.second_q

Qiskit Nature requires a computational chemistry program or library, accessed via a
chemistry *driver*, to be installed on the system for the electronic-structure computation of a
given molecule. A driver is created with a molecular configuration, passed in the format compatible
with that particular driver. This allows custom configuration specific to each computational
chemistry program or library to be passed.

Qiskit Nature thus allows the user to configure a chemistry problem in a way that a chemist already
using the underlying chemistry program or library will be familiar with. The driver is used to
compute some intermediate data, which later will be used to form the input to an algorithm. The
intermediate data is stored in a :class:`~qiskit_nature.properties.GroupedProperty` which in turn
contains multiple :class:`~qiskit_nature.properties.Property` objects.
Some examples for the electronic structure case include:

1. :class:`~qiskit_nature.second_q.operator_factories.electronic.ElectronicEnergy`
2. :class:`~qiskit_nature.second_q.operator_factories.electronic.ParticleNumber`
3. :class:`~qiskit_nature.second_q.operator_factories.electronic.AngularMomentum`
4. :class:`~qiskit_nature.second_q.operator_factories.electronic.Magnetization`
5. :class:`~qiskit_nature.second_q.operator_factories.electronic.ElectronicDipoleMoment`

Once extracted, the structure of this intermediate data is independent of the driver that was
used to compute it.  However the values and level of accuracy of such data will depend on the
underlying chemistry program or library used by the specific driver.

If you want to serialize your input data in order to reuse the same input data in the future or
exchange input data with another person or computer, you have to (until Qiskit Nature 0.3.0) resort
to using the deprecated drivers from the :class:`~qiskit_nature.drivers` module which still output a
:class:`~qiskit_nature.drivers.QMolecule` object. This object can in turn by stored in a binary
format known as the `Hierarchical Data Format 5 (HDF5) <https://support.hdfgroup.org/HDF5/>`__.
You can use the :class:`~qiskit_nature.drivers.second_q.HDF5Driver` to read such a binary
file and directly construct a :class:`~qiskit_nature.properties.GroupedProperty` as you would with
the updated drivers.

In the future, the :class:`~qiskit_nature.properties` module will support some serialization format
directly without the need to fall back onto the deprecated :class:`~qiskit_nature.drivers.QMolecule`
object.

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
   BasisType
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

   qiskit_nature.drivers.second_q.gaussiand
   qiskit_nature.drivers.second_q.psi4d
   qiskit_nature.drivers.second_q.pyquanted
   qiskit_nature.drivers.second_q.pyscfd

The :class:`HDF5Driver` reads molecular data from a pre-existing HDF5 file, as saved from a
:class:`~qiskit_nature.drivers.QMolecule`, and is not dependent on any external chemistry
program/library and needs no special install.

The :class:`FCIDumpDriver` likewise reads from a pre-existing file in this case a standard
FCIDump file and again needs no special install.

Electronic Structure Drivers
============================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ElectronicStructureMoleculeDriver
   ElectronicStructureDriverType
   GaussianDriver
   PSI4Driver
   PyQuanteDriver
   PySCFDriver
   HDF5Driver
   FCIDumpDriver

Vibrational Structure Drivers
=============================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   VibrationalStructureMoleculeDriver
   VibrationalStructureDriverType
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

from .electronic_structure_molecule_driver import (
    ElectronicStructureMoleculeDriver,
    ElectronicStructureDriverType,
)
from .vibrational_structure_molecule_driver import (
    VibrationalStructureMoleculeDriver,
    VibrationalStructureDriverType,
)
from .base_driver import BaseDriver
from .vibrational_structure_driver import VibrationalStructureDriver
from .electronic_structure_driver import ElectronicStructureDriver, MethodType
from .fcidumpd import FCIDumpDriver
from .gaussiand import GaussianDriver, GaussianLogDriver, GaussianLogResult, GaussianForcesDriver
from .hdf5d import HDF5Driver
from .psi4d import PSI4Driver
from .pyquanted import PyQuanteDriver, BasisType
from .pyscfd import PySCFDriver, InitialGuess

__all__ = [
    "ElectronicStructureMoleculeDriver",
    "ElectronicStructureDriverType",
    "VibrationalStructureMoleculeDriver",
    "VibrationalStructureDriverType",
    "MethodType",
    "BaseDriver",
    "VibrationalStructureDriver",
    "ElectronicStructureDriver",
    "FCIDumpDriver",
    "GaussianDriver",
    "GaussianForcesDriver",
    "GaussianLogDriver",
    "GaussianLogResult",
    "HDF5Driver",
    "PSI4Driver",
    "BasisType",
    "PyQuanteDriver",
    "PySCFDriver",
    "InitialGuess",
]
