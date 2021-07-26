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

"""The Freeze-Core Reduction interface."""

from typing import List, Optional

from qiskit_nature.deprecation import DeprecatedType, warn_deprecated_same_type_name

from .second_quantization import FreezeCoreTransformer as NewFreezeCoreTransformer


class FreezeCoreTransformer(NewFreezeCoreTransformer):
    """**DEPRECATED**: Please use the `second_quantization` module instead!

    Please use :class:`~qiskit_nature.transformers.second_quantization.FreezeCoreTransformer`
    instead.
    """

    def __init__(
        self,
        freeze_core: bool = True,
        remove_orbitals: Optional[List[int]] = None,
    ) -> None:
        warn_deprecated_same_type_name(
            "0.2.0",
            DeprecatedType.CLASS,
            "FreezeCoreTransformer",
            "from qiskit_nature.transformers.second_quantization as a direct replacement",
        )
        super().__init__(freeze_core, remove_orbitals)
