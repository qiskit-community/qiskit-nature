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
Second-Quantization Problems (:mod:`qiskit_nature.second_q.problems`)
================================================================================

.. currentmodule:: qiskit_nature.second_q.problems


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
from .electronic_structure_problem import ElectronicStructureProblem
from .electronic_structure_result import DipoleTuple, ElectronicStructureResult
from .lattice_model_problem import LatticeModelProblem
from .lattice_model_result import LatticeModelResult
from .vibrational_structure_problem import VibrationalStructureProblem
from .vibrational_structure_result import VibrationalStructureResult

__all__ = [
    "BaseProblem",
    "ElectronicStructureProblem",
    "DipoleTuple",
    "ElectronicStructureResult",
    "LatticeModelProblem",
    "VibrationalStructureProblem",
    "EigenstateResult",
    "VibrationalStructureResult",
    "LatticeModelResult",
]
