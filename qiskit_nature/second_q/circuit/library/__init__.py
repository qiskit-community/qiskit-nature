# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Circuit Library (:mod:`qiskit_nature.second_q.circuit.library`)
===============================================================

A collection of circuits used as building blocks or inputs for algorithms.

.. currentmodule:: qiskit_nature.second_q.circuit.library

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BogoliubovTransform

Initial states
--------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   FermionicGaussianState
   HartreeFock
   SlaterDeterminant
   VSCF

Ansatzes
--------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   UCC
   UCCSD
   PUCCD
   SUCCD
   CHC
   UVCC
   UVCCSD

Ansatz Utilities
----------------

Utilities such as excitation generators for use with the ansatzes.

.. autosummary::
   :toctree:

   ansatzes.utils
"""

from .ansatzes import (
    UCC,
    UCCSD,
    PUCCD,
    SUCCD,
    CHC,
    UVCC,
    UVCCSD,
)

from .initial_states import FermionicGaussianState, HartreeFock, SlaterDeterminant, VSCF

from .bogoliubov_transform import BogoliubovTransform

__all__ = [
    "UCC",
    "UCCSD",
    "PUCCD",
    "SUCCD",
    "HartreeFock",
    "CHC",
    "UVCC",
    "UVCCSD",
    "VSCF",
    "FermionicGaussianState",
    "SlaterDeterminant",
    "BogoliubovTransform",
]

