# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
SecondQuantizedOp Mappers (:mod:`qiskit_nature.mappers.second_quantization`)
============================================================================

.. currentmodule:: qiskit_nature.mappers.second_quantization

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

"""

from .bravyi_kitaev_mapper import BravyiKitaevMapper
from .bksf import BravyiKitaevSuperFastMapper
from .direct_mapper import DirectMapper
from .fermionic_mapper import FermionicMapper
from .jordan_wigner_mapper import JordanWignerMapper
from .linear_mapper import LinearMapper
from .logarithmic_mapper import LogarithmicMapper
from .logarithmic_mapper import EmbedLocation
from .parity_mapper import ParityMapper
from .qubit_mapper import QubitMapper
from .spin_mapper import SpinMapper
from .vibrational_mapper import VibrationalMapper

__all__ = [
    "BravyiKitaevMapper",
    "BravyiKitaevSuperFastMapper",
    "DirectMapper",
    "EmbedLocation",
    "FermionicMapper",
    "JordanWignerMapper",
    "LinearMapper",
    "LogarithmicMapper",
    "ParityMapper",
    "QubitMapper",
    "SpinMapper",
    "VibrationalMapper",
]
