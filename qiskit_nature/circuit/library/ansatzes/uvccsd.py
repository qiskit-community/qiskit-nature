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
The UVCCSD variational form.
"""

from typing import Optional, List

import logging

from qiskit.circuit import QuantumCircuit
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from .uvcc import UVCC

logger = logging.getLogger(__name__)


class UVCCSD(UVCC):
    """The UVCCSD Ansatz.

    This is a convenience subclass of the UVCC Ansatz. For more information refer to :class:`UVCC`.
    """

    def __init__(self, qubit_converter: Optional[QubitConverter] = None,
                 num_modals: Optional[List[int]] = None,
                 reps: int = 1,
                 initial_state: Optional[QuantumCircuit] = None):
        """
        Args:
            qubit_converter: the QubitConverter instance which takes care of mapping a
            :class:`~.SecondQuantizedOp` to a :class:`~.PauliSumOp` as well as performing all
            configured symmetry reductions on it.
            num_modals: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode num_modals = [4,4,4]
            reps: The number of times to repeat the evolved operators.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
        """
        super().__init__(qubit_converter=qubit_converter,
                         num_modals=num_modals,
                         excitations='sd',
                         reps=reps,
                         initial_state=initial_state)
