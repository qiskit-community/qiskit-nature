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
Initial Points (:mod:`qiskit_nature.algorithms.initial_points`)
===============================================================
Utility classes that provide initial points to use with specific ansatzes.

.. currentmodule:: qiskit_nature.algorithms.initial_points

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   InitialPoint
   HFInitialPoint
   MP2InitialPoint
   VSCFInitialPoint

"""

from .initial_point import InitialPoint
from .hf_initial_point import HFInitialPoint
from .mp2_initial_point import MP2InitialPoint
from .vscf_initial_point import VSCFInitialPoint
from ...deprecation import warn_deprecated, DeprecatedType, NatureDeprecationWarning

warn_deprecated(
    "0.5.0",
    old_type=DeprecatedType.PACKAGE,
    old_name="qiskit_nature.algorithms.initial_points",
    new_type=DeprecatedType.PACKAGE,
    new_name="qiskit_nature.second_q.algorithms.initial_points",
    stack_level=3,
    category=NatureDeprecationWarning,
)

__all__ = ["InitialPoint", "HFInitialPoint", "MP2InitialPoint", "VSCFInitialPoint"]
