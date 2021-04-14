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
Circuit Library (:mod:`qiskit_nature.circuit.library`)
======================================================

A collection of circuits used as building blocks or inputs for algorithms.

.. currentmodule:: qiskit_nature.circuit.library

Initial states
==============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   HartreeFock
   VSCF

Ansatzes
========

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

"""

from .ansatzes import (
    EvolvedOperatorAnsatz,
    UCC,
    UCCSD,
    PUCCD,
    SUCCD,
    CHC,
    UVCC,
    UVCCSD,
)

from .initial_states import (
    HartreeFock,
    VSCF
)

__all__ = [
    'EvolvedOperatorAnsatz',
    'UCC',
    'UCCSD',
    'PUCCD',
    'SUCCD',
    'HartreeFock',
    'CHC',
    'UVCC',
    'UVCCSD',
    'VSCF',
]
