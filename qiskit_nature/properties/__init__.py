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
r"""
Properties (:mod:`qiskit_nature.properties`)
============================================

.. currentmodule:: qiskit_nature.properties


Qiskit Nature ships with a library of commonly evaluates ``Property`` objects (or *observables*).

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Property
   GroupedProperty

.. autosummary::
   :toctree:

   second_quantization
"""

from .grouped_property import GroupedProperty
from .property import Property
from ..deprecation import warn_deprecated, DeprecatedType, NatureDeprecationWarning

warn_deprecated(
    "0.5.0",
    old_type=DeprecatedType.PACKAGE,
    old_name="qiskit_nature.properties",
    new_type=DeprecatedType.PACKAGE,
    new_name="qiskit_nature.second_q.properties",
    stack_level=3,
    category=NatureDeprecationWarning,
)

__all__ = [
    "Property",
    "GroupedProperty",
]
