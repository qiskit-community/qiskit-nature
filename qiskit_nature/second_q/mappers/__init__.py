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
Mappers (:mod:`qiskit_nature.second_q.mappers`)
===============================================

.. currentmodule:: qiskit_nature.second_q.mappers

The classes here are used to convert fermionic, vibrational and spin operators to qubit operators.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QubitMapper

FermionicOp Mappers
+++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   FermionicMapper
   BravyiKitaevMapper
   BravyiKitaevSuperFastMapper
   JordanWignerMapper
   ParityMapper


VibrationalOp Mappers
+++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   VibrationalMapper
   DirectMapper


SpinOp Mappers
++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SpinMapper
   LinearMapper
   LogarithmicMapper

Qubit Converter
+++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QubitConverter
"""

from .bksf import BravyiKitaevSuperFastMapper
from .bravyi_kitaev_mapper import BravyiKitaevMapper
from .jordan_wigner_mapper import JordanWignerMapper
from .parity_mapper import ParityMapper
from .linear_mapper import LinearMapper
from .logarithmic_mapper import LogarithmicMapper
from .direct_mapper import DirectMapper
from .qubit_mapper import QubitMapper
from .qubit_converter import QubitConverter
from .fermionic_mapper import FermionicMapper
from .spin_mapper import SpinMapper
from .vibrational_mapper import VibrationalMapper

__all__ = [
    "BravyiKitaevMapper",
    "BravyiKitaevSuperFastMapper",
    "DirectMapper",
    "JordanWignerMapper",
    "ParityMapper",
    "LinearMapper",
    "LogarithmicMapper",
    "QubitConverter",
    "QubitMapper",
]
