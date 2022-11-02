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

from qiskit import QuantumRegister
from qiskit.circuit.library import BlueprintCircuit
from qiskit.opflow import PauliSumOp
from qiskit_nature.second_q.mappers import DirectMapper
from qiskit_nature.second_q.mappers import QubitConverter
from qiskit_nature.second_q.operators import VibrationalOp

logger = logging.getLogger(__name__)


class VSCF(BlueprintCircuit):
    r"""Initial state for vibrational modes.

    Creates an occupation number vector as defined in [1].
    As example, for 2 modes with 4 modals per mode it creates: :math:`|1000 1000\rangle`.

    References:

        [1] Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.
    """

    def __init__(
        self,
        num_modals: list[int] | None = None,
        qubit_converter: QubitConverter | None = None,
    ) -> None:
        """
        Args:
            num_modals: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode num_modals = [4,4,4]
            qubit_converter: a QubitConverter instance. This argument is currently being ignored
                because only a single use-case is supported at the time of release: that of the
                :class:`DirectMapper`. However, for future-compatibility of this functions
                signature, the argument has already been inserted.
        """
        super().__init__()
        self._num_modals = num_modals
        self._qubit_converter = qubit_converter
        self._bitstr: list[bool] | None = None

        self.qubit_converter = (
            QubitConverter(DirectMapper()) if qubit_converter is None else qubit_converter
        )

    @property
    def qubit_converter(self) -> QubitConverter:
        """The qubit converter."""
        return self._qubit_converter

    @qubit_converter.setter
    def qubit_converter(self, conv: QubitConverter) -> None:
        """Sets the qubit converter."""
        self._invalidate()
        if not isinstance(conv.mapper, DirectMapper):
            logger.warning(
                "The only supported `QubitConverter` is one with a `DirectMapper` as the mapper "
                "instance. However you specified %s as an input, which will be ignored until more "
                "variants will be supported.",
                type(conv.mapper),
            )
            conv = QubitConverter(DirectMapper())
        self._qubit_converter = conv
        self._reset_register()

    @property
    def num_modals(self) -> list[int]:
        """The number of modals per mode."""
        return self._num_modals

    @num_modals.setter
    def num_modals(self, num_modals: list[int]) -> None:
        """Sets the number of modals."""
        self._invalidate()
        self._num_modals = num_modals
        self._reset_register()

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the configuration of the VSCF class is valid.
        Args:
            raise_on_failure: Whether to raise on failure.
        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            an ValueError or TypeError is raised.
        Raises:
            ValueError: If the number of modals per mode is not specified.
            ValueError: If any of the number of modals is less than zero.
            ValueError: If the qubit converter is not specified.
        """
        if self.num_modals is None:
            if raise_on_failure:
                raise ValueError("The number of modals cannot be 'None'.")
            return False

        if any(n < 0 for n in self.num_modals):
            if raise_on_failure:
                raise ValueError(
                    f"The number of modals cannot be smaller than 0 was {self.num_modals}."
                )
            return False

        if self.qubit_converter is None:
            if raise_on_failure:
                raise ValueError("The qubit converter cannot be `None`.")
            return False

        return True

    def _reset_register(self):
        """Reset the register and recompute the mapped VSCF bitstring."""
        self.qregs = []
        self._bitstr = None

        if self._check_configuration(raise_on_failure=False):
            self._bitstr = vscf_bitstring_mapped(self.num_modals, self.qubit_converter)
            self.qregs = [QuantumRegister(len(self._bitstr), name="q")]

    def _build(self) -> None:
        """
        Construct the VSCF initial state given its parameters.
        Returns:
            QuantumCircuit: a quantum circuit preparing the VSCF initial state
            given a number of modals per mode and a qubit converter.
        """
        if self._is_built:
            return

        super()._build()

        # Construct the circuit for bitstring. Since this is defined as an initial state
        # circuit its assumed that this is applied first to the qubits that are initialized to
        # the zero state. Hence we just need to account for all True entries and set those.
        if self._bitstr is not None:
            for i, bit in enumerate(self._bitstr):
                if bit:
                    self.x(i)


def vscf_bitstring_mapped(
    num_modals: list[int],
    qubit_converter: QubitConverter,
) -> list[bool]:
    """Compute the bitstring representing the mapped VSCF initial state
    based on the given the number of modals per mode and qubit converter.

    Args:
        num_modals: A list defining the number of modals per mode. E.g. for a 3 modes system
            with 4 modals per mode num_modals = [4,4,4].
        qubit_converter: A QubitConverter instance.

    Returns:
        The bitstring representing the mapped state of the VSCF initial state as array of bools.
    """

    # get the bitstring encoding initial state
    bitstr = vscf_bitstring(num_modals)

    # encode the bitstring in a `VibrationalOp`
    bitstr_op = VibrationalOp(
        {
            " ".join(
                f"+_{VibrationalOp.build_dual_index(num_modals, idx)}"
                for idx, bit in enumerate(bitstr)
                if bit
            ): 1.0
        },
        num_modals=num_modals,
    )
    # map the `VibrationalOp` to a qubit operator
    qubit_op: PauliSumOp = qubit_converter.convert_match(bitstr_op, check_commutes=False)

    # We check the mapped operator `x` part of the paulis because we want to have particles
    # i.e. True, where the initial state introduced a creation (`+`) operator.
    bits = []
    for bit in qubit_op.primitive.paulis.x[0]:
        bits.append(bit)

    return bits


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
