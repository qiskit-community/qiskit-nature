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
Qiskit Nature Drivers (:mod:`qiskit_nature.drivers`)
====================================================

.. currentmodule:: qiskit_nature.drivers

Driver Common
=============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Molecule
   UnitsType

.. autosummary::
   :toctree:

   second_quantization

"""

from importlib import import_module
from warnings import warn

from .molecule import Molecule
from .units_type import UnitsType

deprecated_names = [
    "HFMethodType",
    "QMolecule",
    "WatsonHamiltonian",
    "BaseDriver",
    "BosonicDriver",
    "FermionicDriver",
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


def __getattr__(name):
    if name in deprecated_names:
        warn(
            f"{name} has been moved to {__name__}.second_quantization.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module(".second_quantization", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(__all__ + deprecated_names)


__all__ = [
    "Molecule",
    "UnitsType",
]
