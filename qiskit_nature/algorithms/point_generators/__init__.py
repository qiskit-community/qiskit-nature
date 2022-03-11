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
Point Generators
(:mod:`qiskit_nature.algorithms.point_generators`)
====================================================================================

Algorithms that can compute initial points to use with particular ansatzes.

.. currentmodule:: qiskit_nature.algorithms.point_generators

The initial point generator interface.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   PointGenerator

The MP2 initial point generator.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MP2PointGenerator

"""

from .point_generator import PointGenerator
from .mp2_point_generator import MP2PointGenerator

__all__ = ["PointGenerator", "MP2PointGenerator"]
