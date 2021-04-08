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
SecondQuantizedOp Mappers (:mod:`qiskit_nature.converters.second_quantization`)
===============================================================================

.. currentmodule:: qiskit_nature.converters.second_quantization

The classes here are used to convert fermionic, vibrational and spin operators to qubit operators,
using mappers and other techniques that can also reduce the problem such as leveraging
Z2 Symmetries.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QubitConverter

"""

from .qubit_converter import QubitConverter

__all__ = [
    'QubitConverter',
]
