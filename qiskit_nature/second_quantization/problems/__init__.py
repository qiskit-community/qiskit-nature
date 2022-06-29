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
Second-Quantization Problems (:mod:`qiskit_nature.second_quantization.problems`)
================================================================================

.. currentmodule:: qiskit_nature.second_quantization.problems


.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseProblem
   ElectronicStructureProblem
   VibrationalStructureProblem
   LatticeModelProblem

Submodules
==========
.. autosummary::
   :toctree: ../stubs/

   lattice

"""

from .base_problem import BaseProblem, EigenstateResult
from .electronic import ElectronicStructureProblem, ElectronicStructureResult
from .lattice import LatticeModelProblem, LatticeModelResult
from .vibrational import VibrationalStructureProblem, VibrationalStructureResult


__all__ = [
    "BaseProblem",
    "ElectronicStructureProblem",
    "LatticeModelProblem",
    "VibrationalStructureProblem",
    "EigenstateResult",
    "ElectronicStructureResult",
    "VibrationalStructureResult",
    "LatticeModelResult",
]
