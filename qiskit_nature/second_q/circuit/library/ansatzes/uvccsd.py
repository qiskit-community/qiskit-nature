# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
The UVCCSD Ansatz.
"""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.mappers import QubitMapper
from .uvcc import UVCC


class UVCCSD(UVCC):
    """The UVCCSD Ansatz.

    This is a convenience subclass of the UVCC ansatz. For more information refer to :class:`UVCC`.
    """

    def __init__(
        self,
        num_modals: list[int] | None = None,
        qubit_mapper: QubitMapper | None = None,
        *,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
    ) -> None:
        # pylint: disable=unused-argument
        """
        Args:
            num_modals: A list defining the number of modals per mode. E.g. for a 3-mode system
                with 4 modals per mode ``num_modals = [4, 4, 4]``.
            qubit_mapper: The :class:`~qiskit_nature.second_q.mappers.QubitMapper` which takes care
                of mapping to a qubit operator.
            reps: The number of times to repeat the evolved operators.
            initial_state: A ``QuantumCircuit`` object to prepend to the circuit.
        """
        super().__init__(
            num_modals=num_modals,
            excitations="sd",
            qubit_mapper=qubit_mapper,
            reps=reps,
            initial_state=initial_state,
        )
