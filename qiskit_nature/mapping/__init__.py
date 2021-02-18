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

"""The Qubit Mappings."""

from qiskit_nature.mapping.mappings.bravyi_kitaev_mapping import BravyiKitaevMapping
from qiskit_nature.mapping.mappings.bravyi_kitaev_super_fast_mapping import \
    BravyiKitaevSuperFastMapping
from qiskit_nature.mapping.mappings.direct_mapping import DirectMapping
from qiskit_nature.mapping.mappings.jordan_wigner_mapping import JordanWignerMapping
from qiskit_nature.mapping.mappings.linear_mapping import LinearMapping
from qiskit_nature.mapping.mappings.logarithmic_mapping import LogarithmicMapping
from qiskit_nature.mapping.mappings.parity_mapping import ParityMapping
from qiskit_nature.mapping.mappings.qubit_mapping import QubitMapping

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
