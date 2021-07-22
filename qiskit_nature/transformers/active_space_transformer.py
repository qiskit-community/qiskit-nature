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

"""The Active-Space Reduction interface."""

from typing import List, Optional, Tuple, Union

from qiskit_nature.deprecation import DeprecatedType, warn_deprecated_same_type_name

from .second_quantization import ActiveSpaceTransformer as NewActiveSpaceTransformer


# inherited from super-class
# pylint: disable=missing-class-docstring
class ActiveSpaceTransformer(NewActiveSpaceTransformer):
    def __init__(
        self,
        num_electrons: Optional[Union[int, Tuple[int, int]]] = None,
        num_molecular_orbitals: Optional[int] = None,
        active_orbitals: Optional[List[int]] = None,
    ) -> None:
        warn_deprecated_same_type_name(
            "0.2.0",
            DeprecatedType.CLASS,
            "ActiveSpaceTransformer",
            "from qiskit_nature.transformers.second_quantization as a direct replacement",
        )
        super().__init__(num_electrons, num_molecular_orbitals, active_orbitals)
