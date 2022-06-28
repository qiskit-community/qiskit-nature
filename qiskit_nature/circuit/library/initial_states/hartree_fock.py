# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Hartree-Fock initial state."""

from __future__ import annotations

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.utils.validation import validate_min

from qiskit_nature.second_quantization.operators import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.second_quantization.operators.fermionic import BravyiKitaevSuperFastMapper


class HartreeFock(QuantumCircuit):
    """A Hartree-Fock initial state."""

    def __init__(
        self,
        num_spin_orbitals: int,
        num_particles: tuple[int, int],
        qubit_converter: QubitConverter,
    ) -> None:
        """
        Args:
            num_spin_orbitals: The number of spin orbitals, has a min. value of 1.
            num_particles: The number of particles as a tuple storing the number of alpha- and
                           beta-spin electrons in the first and second number, respectively.
            qubit_converter: a QubitConverter instance.

        Raises:
            TypeError: If qubit_converter contains BravyiKitaevSuperFastMapper. See
                https://github.com/Qiskit/qiskit-nature/issues/537 for more information.
        """
        if isinstance(qubit_converter.mapper, BravyiKitaevSuperFastMapper):
            raise TypeError(
                "Unsupported mapper in qubit_converter: ",
                type(qubit_converter.mapper),
                ". See https://github.com/Qiskit/qiskit-nature/issues/537",
            )
        # Get the mapped/tapered hartree fock bitstring as we need it to match to whatever
        # conversion was done by the given qubit converter
        bitstr = hartree_fock_bitstring_mapped(
            num_spin_orbitals, num_particles, qubit_converter, True
        )
        # Construct the circuit for this bitstring. Since this is defined as an initial state
        # circuit its assumed that this is applied first to the qubits that are initialized to
        # the zero state. Hence we just need to account for all True entries and set those.
        qr = QuantumRegister(len(bitstr), "q")
        super().__init__(qr, name="HF")

        for i, bit in enumerate(bitstr):
            if bit:
                self.x(i)


def hartree_fock_bitstring_mapped(
    num_spin_orbitals: int,
    num_particles: tuple[int, int],
    qubit_converter: QubitConverter,
    match_convert: bool = True,
) -> list[bool]:
    """Compute the bitstring representing the mapped Hartree-Fock state for the specified system.

    Args:
        num_spin_orbitals: The number of spin orbitals, has a min. value of 1.
        num_particles: The number of particles as a tuple (alpha, beta) containing the number of
            alpha- and  beta-spin electrons, respectively.
        qubit_converter: A QubitConverter instance.
        match_convert: Whether to use `convert_match` method of the qubit converter (default),
            or just do mapping and possibly two qubit reduction but no tapering. The latter
            is an advanced usage - e.g. if we are trying to auto-select the tapering sector
            then we would not want any match conversion done on a converter that was set to taper.

    Returns:
        The bitstring representing the mapped state of the Hartree-Fock state as array of bools.
    """

    # get the bitstring encoding the Hartree Fock state
    bitstr = hartree_fock_bitstring(num_spin_orbitals, num_particles)

    # encode the bitstring as a `FermionicOp`
    label = ["+" if bit else "I" for bit in bitstr]
    bitstr_op = FermionicOp("".join(label), display_format="sparse")

    # map the `FermionicOp` to a qubit operator
    qubit_op: PauliSumOp = (
        qubit_converter.convert_match(bitstr_op, check_commutes=False)
        if match_convert
        else qubit_converter.convert_only(bitstr_op, num_particles)
    )

    # We check the mapped operator `x` part of the paulis because we want to have particles
    # i.e. True, where the initial state introduced a creation (`+`) operator.
    bits = []
    for bit in qubit_op.primitive.paulis.x[0]:
        bits.append(bit)

    return bits


def hartree_fock_bitstring(num_spin_orbitals: int, num_particles: tuple[int, int]) -> list[bool]:
    """Compute the bitstring representing the Hartree-Fock state for the specified system.

    Args:
        num_spin_orbitals: The number of spin orbitals, has a min. value of 1.
        num_particles: The number of particles as a tuple storing the number of alpha- and beta-spin
                       electrons in the first and second number, respectively.

    Returns:
        The bitstring representing the state of the Hartree-Fock state as array of bools.

    Raises:
        ValueError: If the total number of particles is larger than the number of orbitals.
    """
    # validate the input
    validate_min("num_spin_orbitals", num_spin_orbitals, 1)
    num_alpha, num_beta = num_particles

    if sum(num_particles) > num_spin_orbitals:
        raise ValueError("# of particles must be less than or equal to # of orbitals.")

    half_orbitals = num_spin_orbitals // 2
    bitstr = np.zeros(num_spin_orbitals, bool)
    bitstr[:num_alpha] = True
    bitstr[half_orbitals : (half_orbitals + num_beta)] = True

    return bitstr.tolist()
