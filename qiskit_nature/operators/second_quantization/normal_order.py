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

"""Data class for normal order."""

from dataclasses import dataclass
from typing import List


@dataclass
class NormalOrder:
    """Data class for normal order.
    `NormalOrder` holds the normal order position of the creation and annihilation operators
    in little endian.
    """

    plus_position: List[int]
    minus_position: List[int]
    coeff: complex
