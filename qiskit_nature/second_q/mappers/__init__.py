# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
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

The classes here are used to convert fermionic, bosonic, vibrational and spin operators to qubit
operators.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QubitMapper
   ModeBasedMapper

FermionicOp Mappers
+++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BravyiKitaevMapper
   BravyiKitaevSuperFastMapper
   JordanWignerMapper
   ParityMapper
   TernaryTreeMapper

**Note:** :class:`~qiskit_nature.second_q.mappers.TernaryTreeMapper` maps
:class:`~qiskit_nature.second_q.operators.MajoranaOp` to
:class:`~qiskit.quantum_info.SparsePauliOp`. In order to use it on a
:class:`~qiskit_nature.second_q.operators.FermionicOp`, convert to a
:class:`~qiskit_nature.second_q.operators.MajoranaOp` first using
:code:`MajoranaOp.from_fermionic_op()`.

**Interleaved Qubit-Ordering:** If you want to generate qubit operators where the alpha-spin and
beta-spin components are mapped to the qubit register in an interleaved (rather than the default
blocked) order, you can use the following wrapper:

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   InterleavedQubitMapper


BosonicOp Mappers
+++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BosonicLinearMapper

VibrationalOp Mappers
+++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   DirectMapper


SpinOp Mappers
++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   LinearMapper
   LogarithmicMapper

Tapered Qubit Mapper
++++++++++++++++++++

If you want to make use of the symmetries of your problem and add a step of tapering
after the mapping to qubit operators, you can use the following wrapper for symmetry reduction:

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   TaperedQubitMapper

"""

from .bksf import BravyiKitaevSuperFastMapper
from .bravyi_kitaev_mapper import BravyiKitaevMapper
from .jordan_wigner_mapper import JordanWignerMapper
from .parity_mapper import ParityMapper
from .ternary_tree_mapper import TernaryTreeMapper
from .linear_mapper import LinearMapper
from .bosonic_linear_mapper import BosonicLinearMapper
from .logarithmic_mapper import LogarithmicMapper
from .direct_mapper import DirectMapper
from .qubit_mapper import QubitMapper
from .mode_based_mapper import ModeBasedMapper
from .interleaved_qubit_mapper import InterleavedQubitMapper
from .tapered_qubit_mapper import TaperedQubitMapper

__all__ = [
    "BravyiKitaevMapper",
    "BravyiKitaevSuperFastMapper",
    "DirectMapper",
    "JordanWignerMapper",
    "ParityMapper",
    "TernaryTreeMapper",
    "LinearMapper",
    "BosonicLinearMapper",
    "LogarithmicMapper",
    "QubitMapper",
    "InterleavedQubitMapper",
    "TaperedQubitMapper",
    "ModeBasedMapper",
]
