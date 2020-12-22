# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Qubit Mappings."""

from .bravyi_kitaev_mapping import BravyiKitaevMapping
from .bravyi_kitaev_super_fast_mapping import BravyiKitaevSuperFastMapping
from .direct_mapping import DirectMapping
from .jordan_wigner_mapping import JordanWignerMapping
from .linear_mapping import LinearMapping
from .logarithmic_mapping import LogarithmicMapping
from .parity_mapping import ParityMapping
from .qubit_mapping import QubitMapping

__all__ = [
        'BravyiKitaevMapping',
        'BravyiKitaevSuperFastMapping',
        'DirectMapping',
        'JordanWignerMapping',
        'LinearMapping',
        'LogarithmicMapping',
        'ParityMapping',
        'QubitMapping',
    ]
