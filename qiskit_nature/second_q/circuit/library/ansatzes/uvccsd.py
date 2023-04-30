# This code is part of Qiskit.
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
from qiskit_nature.deprecation import deprecate_arguments
from qiskit_nature.second_q.mappers import QubitConverter, QubitMapper
from .uvcc import UVCC


class UVCCSD(UVCC):
    """The UVCCSD Ansatz.

    This is a convenience subclass of the UVCC ansatz. For more information refer to :class:`UVCC`.
    """

    @deprecate_arguments(
        "0.6.0",
        {"qubit_converter": "qubit_mapper"},
        additional_msg=(
            ". Additionally, the QubitConverter type in the qubit_mapper argument is deprecated "
            "and support for it will be removed together with the qubit_converter argument."
        ),
    )
    def __init__(
        self,
        num_modals: list[int] | None = None,
        qubit_mapper: QubitConverter | QubitMapper | None = None,
        *,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        qubit_converter: QubitConverter | QubitMapper | None = None,
    ) -> None:
        # pylint: disable=unused-argument
        """
        Args:
            num_modals: A list defining the number of modals per mode. E.g. for a 3-mode system
                with 4 modals per mode ``num_modals = [4, 4, 4]``.
            qubit_mapper: The :class:`~qiskit_nature.second_q.mappers.QubitMapper` or
                :class:`~qiskit_nature.second_q.mappers.QubitConverter` instance (use of the latter
                is deprecated) which takes care of mapping to a qubit operator.
            reps: The number of times to repeat the evolved operators.
            initial_state: A ``QuantumCircuit`` object to prepend to the circuit.
            qubit_converter: DEPRECATED The :class:`~qiskit_nature.second_q.mappers.QubitConverter`
                or :class:`~qiskit_nature.second_q.mappers.QubitMapper` instance which takes care of
                mapping to a qubit operator.
        """
        super().__init__(
            num_modals=num_modals,
            excitations="sd",
            qubit_mapper=qubit_mapper,
            reps=reps,
            initial_state=initial_state,
        )
