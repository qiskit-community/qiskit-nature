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
Transformers (:mod:`qiskit_nature.transformers`)
================================================

.. currentmodule:: qiskit_nature.transformers

.. autosummary::
   :toctree:

   second_quantization

"""

from importlib import import_module
from ..deprecation import DeprecatedType, warn_deprecated_same_type_name

deprecated_names = [
    "ActiveSpaceTransformer",
    "BaseTransformer",
    "FreezeCoreTransformer",
]


def __getattr__(name):
    if name in deprecated_names:
        warn_deprecated_same_type_name(
            "0.2.0",
            DeprecatedType.PROPERTY,
            name,
            f"from {__name__}.second_quantization as a direct replacement",
        )
        module = import_module(".second_quantization", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(deprecated_names)
