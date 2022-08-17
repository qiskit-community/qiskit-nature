# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Excited State Solving Algorithms (:mod:`qiskit_nature.algorithms.excited_states_solvers`)
=========================================================================================

.. currentmodule:: qiskit_nature.algorithms.excited_states_solvers

.. autosummary::
   :toctree: ../stubs/

   eigensolver_factories

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ExcitedStatesEigensolver
   QEOM
"""

from .excited_states_solver import ExcitedStatesSolver
from .qeom import QEOM
from .eigensolver_factories import EigensolverFactory, NumPyEigensolverFactory
from .excited_states_eigensolver import ExcitedStatesEigensolver
from ...deprecation import warn_deprecated, DeprecatedType, NatureDeprecationWarning

warn_deprecated(
    "0.5.0",
    old_type=DeprecatedType.PACKAGE,
    old_name="qiskit_nature.algorithms.excited_state_solvers",
    new_type=DeprecatedType.PACKAGE,
    new_name="qiskit_nature.second_q.algorithms.excited_state_solvers",
    stack_level=3,
    category=NatureDeprecationWarning,
)

__all__ = [
    "ExcitedStatesSolver",
    "ExcitedStatesEigensolver",
    "EigensolverFactory",
    "NumPyEigensolverFactory",
    "QEOM",
]
