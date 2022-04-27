# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
====================================================================================
Initial Points
(:mod:`qiskit_nature.algorithms.initial_points`)
====================================================================================

Algorithms that can compute initial points to use with particular ansatzes.

.. currentmodule:: qiskit_nature.algorithms.initial_points

The initial point generator interface.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   InitialPoint

The MP2 initial point generator.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MP2InitialPoint

"""

from .initial_point import InitialPoint
from .mp2_initial_point import MP2InitialPoint

__all__ = ["InitialPoint", "MP2InitialPoint"]
