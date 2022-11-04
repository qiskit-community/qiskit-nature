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
Problems (:mod:`qiskit_nature.second_q.problems`)
=====================================================================

.. currentmodule:: qiskit_nature.second_q.problems

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseProblem

   ElectronicBasis
   VibrationalBasis
   HarmonicBasis

   ElectronicStructureProblem
   VibrationalStructureProblem
   LatticeModelProblem

   PropertiesContainer
   ElectronicPropertiesContainer
   VibrationalPropertiesContainer
   LatticePropertiesContainer

   EigenstateResult
   ElectronicStructureResult
   VibrationalStructureResult
   LatticeModelResult
"""

from .base_problem import BaseProblem, EigenstateResult
from .properties_container import PropertiesContainer

from .electronic_basis import ElectronicBasis
from .electronic_properties_container import ElectronicPropertiesContainer
from .electronic_structure_problem import ElectronicStructureProblem
from .electronic_structure_result import DipoleTuple, ElectronicStructureResult

from .lattice_model_problem import LatticeModelProblem
from .lattice_model_result import LatticeModelResult
from .lattice_properties_container import LatticePropertiesContainer

from .harmonic_basis import HarmonicBasis
from .vibrational_basis import VibrationalBasis
from .vibrational_properties_container import VibrationalPropertiesContainer
from .vibrational_structure_problem import VibrationalStructureProblem
from .vibrational_structure_result import VibrationalStructureResult

__all__ = [
    "BaseProblem",
    "EigenstateResult",
    "PropertiesContainer",
    "ElectronicBasis",
    "ElectronicStructureProblem",
    "ElectronicStructureResult",
    "ElectronicPropertiesContainer",
    "DipoleTuple",
    "LatticeModelProblem",
    "LatticeModelResult",
    "LatticePropertiesContainer",
    "VibrationalBasis",
    "HarmonicBasis",
    "VibrationalPropertiesContainer",
    "VibrationalStructureProblem",
    "VibrationalStructureResult",
]
