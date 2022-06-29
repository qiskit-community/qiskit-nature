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
Results (:mod:`qiskit_nature.results`)
======================================

.. currentmodule:: qiskit_nature.results

Qiskit Nature results such as for electronic and vibrational structure. Algorithms
may extend these to provide algorithm specific aspects in their result.


Results
=======

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BOPESSamplerResult
   ProteinFoldingResult

Protein Folding Result support classes
--------------------------------------

.. autosummary::
   :toctree:

   utils

"""

from .bopes_sampler_result import BOPESSamplerResult
from .protein_folding_result import ProteinFoldingResult

__all__ = [
    "BOPESSamplerResult",
    "ProteinFoldingResult",
]
