# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Initial state for vibrational modes."""

from __future__ import annotations

import logging

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit_nature.mappers.second_quantization import DirectMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.operators.second_quantization import VibrationalOp

logger = logging.getLogger(__name__)


class VSCF(QuantumCircuit):
    r"""Initial state for vibrational modes.

    Creates an occupation number vector as defined in [1].
    As example, for 2 modes with 4 modals per mode it creates: :math:`|1000 1000\rangle`.

    References:

        [1] Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.
    """

    def __init__(
        self,
        num_modals: list[int],
        qubit_converter: QubitConverter | None = None,
    ) -> None:
        """
        Args:
            num_modals: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode num_modals = [4,4,4]
            qubit_converter: a QubitConverter instance. This argument is currently being ignored
                             because only a single use-case is supported at the time of release:
                             that of the :class:`DirectMapper`. However, for future-compatibility of
                             this functions signature, the argument has already been inserted.
        """
        # get the bitstring encoding initial state
        bitstr = vscf_bitstring(num_modals)

        # encode the bitstring in a `VibrationalOp`
        label = ["+" if bit else "I" for bit in bitstr]
        bitstr_op = VibrationalOp("".join(label), num_modes=len(num_modals), num_modals=num_modals)

        # map the `VibrationalOp` to a qubit operator
        if qubit_converter is not None:
            logger.warning(
                "The only supported `QubitConverter` is one with a `DirectMapper` as the mapper "
                "instance. However you specified %s as an input, which will be ignored until more "
                "variants will be supported.",
                str(qubit_converter),
            )
        qubit_converter = QubitConverter(DirectMapper())
        qubit_op: PauliSumOp = qubit_converter.convert_match(bitstr_op)

        # construct the circuit
        qr = QuantumRegister(qubit_op.num_qubits, "q")
        super().__init__(qr, name="VSCF")

        # add gates in the right positions
        for i, bit in enumerate(qubit_op.primitive.paulis.x[0]):
            if bit:
                self.x(i)


def vscf_bitstring(num_modals: list[int]) -> list[bool]:
    """Compute the bitstring representing the VSCF initial state based on the modals per mode.

    Args:
        num_modals: Is a list defining the number of modals per mode. E.g. for a 3 modes system
            with 4 modals per mode num_modals = [4,4,4].

    Returns:
        The bitstring representing the state of the VSCF state as array of bools.
    """
    num_qubits = sum(num_modals)
    bitstr = np.zeros(num_qubits, bool)
    count = 0
    for modal in num_modals:
        bitstr[count] = True
        count += modal

    return bitstr.tolist()
