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

from typing import Union, List, Tuple
import logging
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.utils.validation import validate_min

from qiskit_nature.operators.second_quantization import FermionicOp, SecondQuantizedOp
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter

logger = logging.getLogger(__name__)


class HartreeFock(QuantumCircuit):
    """A Hartree-Fock initial state."""

    def __init__(self,
                 num_orbitals: int,
                 num_particles: Union[Tuple[int, int], int],
                 qubit_converter: QubitConverter) -> None:
        """
        Args:
            num_orbitals: The number of spin orbitals, has a min. value of 1.
            num_particles: The number of particles. If this is an integer, it is the total (even)
                number of particles. If a tuple, the first number is alpha and the second number is
                beta.
            qubit_converter: a QubitConverter instance.
        """

        # get the bitstring encoding the Hartree Fock state
        bitstr_op = hartree_fock_bitstring(num_orbitals, num_particles)

        qubit_op = qubit_converter.to_qubit_ops([bitstr_op])[0]

        # construct the circuit
        qr = QuantumRegister(qubit_op.num_qubits, 'q')
        super().__init__(qr, name='HF')

        # add gates in the right positions
        for i, bit in enumerate(qubit_op.primitive.table.X[0]):
            if bit:
                self.x(i)


def hartree_fock_bitstring(num_orbitals: int,
                           num_particles: Union[Tuple[int, int], int]) -> List[bool]:
    """Compute the bitstring representing the Hartree-Fock state for the specified system.

    Args:
        num_orbitals: The number of spin orbitals, has a min. value of 1.
        num_particles: The number of particles. If this is an integer, it is the total (even) number
            of particles. If a tuple, the first number is alpha and the second number is beta.

    Returns:
        The bitstring representing the state of the Hartree-Fock state as array of bools.

    Raises:
        ValueError: If the total number of particles is larger than the number of orbitals.
    """
    # validate the input
    validate_min('num_orbitals', num_orbitals, 1)

    if isinstance(num_particles, tuple):
        num_alpha, num_beta = num_particles
    else:
        logger.info('We assume that the number of alphas and betas are the same.')
        num_alpha = num_beta = num_particles // 2

    num_particles = num_alpha + num_beta

    if num_particles > num_orbitals:
        raise ValueError('# of particles must be less than or equal to # of orbitals.')

    half_orbitals = num_orbitals // 2
    bitstr = np.zeros(num_orbitals, bool)
    bitstr[:num_alpha] = True
    bitstr[half_orbitals:(half_orbitals + num_beta)] = True

    label = ['+' if bit else 'I' for bit in bitstr]

    bitstr_op = SecondQuantizedOp([FermionicOp(''.join(label))])
    return bitstr_op
