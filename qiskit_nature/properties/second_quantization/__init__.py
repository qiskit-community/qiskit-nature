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
r"""
Second-Quantization Properties (:mod:`qiskit_nature.properties.second_quantization`)
====================================================================================

.. currentmodule:: qiskit_nature.properties.second_quantization


These objects provide the means to map from raw data (as produced by e.g. a
``qiskit_nature.drivers.BaseDriver``) to an operator which can be evaluated during the Quantum
algorithm.
This provides a powerful interface through which a user will be able to insert custom objects which
they desire to evaluate.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SecondQuantizedProperty

.. autosummary::
   :toctree:

   electronic
   vibrational
"""

from .second_quantized_property import SecondQuantizedProperty

__all__ = [
    "SecondQuantizedProperty",
]
