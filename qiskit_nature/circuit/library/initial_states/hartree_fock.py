# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Hartree-Fock initial state."""

from typing import List, Tuple

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.utils.validation import validate_min

from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter

# TODO: There are still some open questions on the exact design/implementation of this class, which
# can only be resolved once we finalize the QubitConverter and implement the Z2 symmetry reduction
# logic into there.
# The reason being, that we would like to be able to use the HF bit-string as an indicator of the Z2
# symmetry sector in which the ground state lies. In the past, this is what hartree_fock_bitstring
# was used for, but as of now, it does not support that.
# If we want to support that, we must be able to use the QubitConverter in such a way that it
# performs the mapping as well as the two_qubit_reduction (if specified) but not the tapering, when
# trying to find the sector of the ground state.
# However, when actually constructing the initial state to be used in a circuit, the symmetry
# reductions need to be applied.
#
# We are leaving the exact implementation of this open for now, because the final API of the
# QubitConverter is not clear yet. As of now, this implementation works as intended for all
# purposes, because Z2 symmetries are not yet supported by the QubitConverter at all, anyways.


class HartreeFock(QuantumCircuit):
    """A Hartree-Fock initial state."""

    def __init__(self,
                 num_spin_orbitals: int,
                 num_particles: Tuple[int, int],
                 qubit_converter: QubitConverter) -> None:
        """
        Args:
            num_spin_orbitals: The number of spin orbitals, has a min. value of 1.
            num_particles: The number of particles as a tuple storing the number of alpha- and
                           beta-spin electrons in the first and second number, respectively.
            qubit_converter: a QubitConverter instance.
        """

        # get the bitstring encoding the Hartree Fock state
        bitstr = hartree_fock_bitstring(num_spin_orbitals, num_particles)

        # encode the bitstring as a `FermionicOp`
        label = ['+' if bit else 'I' for bit in bitstr]
        bitstr_op = FermionicOp(''.join(label))

        # map the `FermionicOp` to a qubit operator
        qubit_op: PauliSumOp = qubit_converter.convert_match(bitstr_op)

        # construct the circuit
        qr = QuantumRegister(qubit_op.num_qubits, 'q')
        super().__init__(qr, name='HF')

        # Add gates in the right positions: we are only interested in the `X` gates because we want
        # to create particles (0 -> 1) where the initial state introduced a creation (`+`) operator.
        for i, bit in enumerate(qubit_op.primitive.table.X[0]):
            if bit:
                self.x(i)


def hartree_fock_bitstring(num_spin_orbitals: int,
                           num_particles: Tuple[int, int]) -> List[bool]:
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
    validate_min('num_spin_orbitals', num_spin_orbitals, 1)
    num_alpha, num_beta = num_particles

    if sum(num_particles) > num_spin_orbitals:
        raise ValueError('# of particles must be less than or equal to # of orbitals.')

    half_orbitals = num_spin_orbitals // 2
    bitstr = np.zeros(num_spin_orbitals, bool)
    bitstr[:num_alpha] = True
    bitstr[half_orbitals:(half_orbitals + num_beta)] = True

    return bitstr.tolist()
