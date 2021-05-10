# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Qiskit Nature Runtime (:mod:`qiskit_nature.runtime`)
====================================================

.. currentmodule:: qiskit_nature.runtime

Programs that embed Qiskit Runtime in the algorithmic interfaces and faciliate usage of
algorithms and scripts in the cloud.

.. autosummary::
   :toctree:

   VQEProgram
   VQEProgramResult

"""

from .vqe_program import VQEProgram, VQEProgramResult

__all__ = ["VQEProgram", "VQEProgramResult"]
