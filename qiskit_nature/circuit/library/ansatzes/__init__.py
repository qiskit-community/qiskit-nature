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

"""
QuantumCircuit Ansatzes (:mod:`qiskit_nature.circuit.library.ansatzes`)
=======================================================================

.. currentmodule:: qiskit_nature.circuit.library.ansatzes


Electronic Structure Ansatzes
+++++++++++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   UCC
   UCCSD
   PUCCD
   SUCCD


Vibrational Structure Ansatzes
++++++++++++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   CHC
   UVCC
   UVCCSD

"""

from .chc import CHC
from .evolved_operator_ansatz import EvolvedOperatorAnsatz
from .puccd import PUCCD
from .succd import SUCCD
from .ucc import UCC
from .uccsd import UCCSD
from .uvcc import UVCC
from .uvccsd import UVCCSD

__all__ = [
    'CHC',
    'EvolvedOperatorAnsatz',
    'PUCCD',
    'SUCCD',
    'UCC',
    'UCCSD',
    'UVCC',
    'UVCCSD',
]
