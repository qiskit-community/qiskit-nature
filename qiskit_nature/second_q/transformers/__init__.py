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
Transformers (:mod:`qiskit_nature.second_q.transformers`)
=========================================================================================

.. currentmodule:: qiskit_nature.second_q.transformers

Transformers act on a :class:`~qiskit_nature.drivers.QMolecule` to produce an altered copy of it as
per the specific transformer. So for instance the :class:`FreezeCoreTransformer` will alter the
integrals and number of particles in a way that freezes the core orbitals, storing an extracted
energy in the QMolecule to compensate for this that would need to be included back into any ground
state energy computation to get complete result.

.. autosummary::
   :toctree: ../stubs/

   ActiveSpaceTransformer
   FreezeCoreTransformer

"""

from .active_space_transformer import ActiveSpaceTransformer
from .freeze_core_transformer import FreezeCoreTransformer
from .base_transformer import BaseTransformer

__all__ = ["ActiveSpaceTransformer", "FreezeCoreTransformer", "BaseTransformer"]
