# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
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
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import DirectMapper
from qiskit_nature.second_q.mappers import QubitMapper, TaperedQubitMapper
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
        qubit_mapper: QubitMapper | None = None,
    ) -> None:
        # pylint: disable=unused-argument
        """
        Args:
            num_modals: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode num_modals = [4,4,4]
            qubit_mapper: a QubitMapper. This argument is currently being ignored because only a
                single use-case is supported at the time of release: that of the
                :class:`.DirectMapper`. However, for future-compatibility of this functions
                signature, the argument has already been inserted.
        """
        super().__init__()
        self._num_modals = num_modals
        self._qubit_mapper = qubit_mapper
        self._bitstr: list[bool] | None = None

        self.qubit_mapper = DirectMapper() if qubit_mapper is None else qubit_mapper

    @property
    def qubit_mapper(self) -> QubitMapper | None:
        """The qubit mapper."""
        return self._qubit_mapper

    @qubit_mapper.setter
    def qubit_mapper(self, mapper: QubitMapper | None) -> None:
        """Sets the qubit mapper."""
        self._invalidate()

        if isinstance(mapper, TaperedQubitMapper):
            # we also include the TaperedQubitMapper here, purely for the check done below
            mapper = mapper.mapper

        if not isinstance(mapper, DirectMapper):
            logger.warning(
                "The only supported `QubitMapper` for this application are those based on the "
                "`DirectMapper`. However you specified %s as an input, which will be ignored until "
                "more variants will be supported.",
                type(mapper),
            )
            mapper = DirectMapper()
        self._qubit_mapper = mapper
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
            ValueError: If the qubit mapper is not specified.
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

        if self.qubit_mapper is None:
            if raise_on_failure:
                raise ValueError("The qubit mapper cannot be `None`.")
            return False

        return True

    def _reset_register(self):
        """Reset the register and recompute the mapped VSCF bitstring."""
        self.qregs = []
        self._bitstr = None

        if self._check_configuration(raise_on_failure=False):
            self._bitstr = vscf_bitstring_mapped(self.num_modals, self.qubit_mapper)
            self.qregs = [QuantumRegister(len(self._bitstr), name="q")]

    def _build(self) -> None:
        """
        Construct the VSCF initial state given its parameters.
        Returns:
            QuantumCircuit: a quantum circuit preparing the VSCF initial state
            given a number of modals per mode and a qubit mapper.
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
    qubit_mapper: QubitMapper,
) -> list[bool]:
    # pylint: disable=unused-argument
    """Compute the bitstring representing the mapped VSCF initial state
    based on the given the number of modals per mode and qubit mapper.

    Args:
        num_modals: A list defining the number of modals per mode. E.g. for a 3 modes system
            with 4 modals per mode num_modals = [4,4,4].
        qubit_mapper: A QubitMapper.

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
    qubit_op: SparsePauliOp
    if isinstance(qubit_mapper, TaperedQubitMapper):
        # To avoid checking commutativity, we call the two methods separately.
        qubit_op = qubit_mapper.map_clifford(bitstr_op)
        qubit_op = qubit_mapper.taper_clifford(qubit_op, check_commutes=False)
    else:
        qubit_op = qubit_mapper.map(bitstr_op)

    # We check the mapped operator `x` part of the paulis because we want to have particles
    # i.e. True, where the initial state introduced a creation (`+`) operator.
    bits = []
    for bit in qubit_op.paulis.x[0]:
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
