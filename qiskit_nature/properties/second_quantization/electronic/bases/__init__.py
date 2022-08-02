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
Electronic Bases (:mod:`qiskit_nature.properties.second_quantization.electronic.bases`)
=======================================================================================

.. currentmodule:: qiskit_nature.properties.second_quantization.electronic.bases

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ElectronicBasis
   ElectronicBasisTransform
"""

from .electronic_basis import ElectronicBasis
from .electronic_basis_transform import ElectronicBasisTransform
from .....deprecation import warn_deprecated, DeprecatedType, NatureDeprecationWarning

warn_deprecated(
    "0.5.0",
    old_type=DeprecatedType.PACKAGE,
    old_name="qiskit_nature.properties.second_quantization.electronic.bases",
    new_type=DeprecatedType.PACKAGE,
    new_name="qiskit_nature.second_q.problems.bases",
    category=NatureDeprecationWarning,
)

__all__ = [
    "ElectronicBasis",
    "ElectronicBasisTransform",
]
