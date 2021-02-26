# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Parity Mapper."""

from typing import Union, List
import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli

from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.operators.second_quantization.particle_op import ParticleOp

from .qubit_mapper import QubitMapper


class ParityMapper(QubitMapper):
    """The Parity fermion-to-qubit mapping. """

    def supports_particle_type(self, particle_type: ParticleOp) -> bool:
        """Returns whether the queried particle-type operator is supported by this mapping.

        Args:
            particle_type: the particle-type to query support for.

        Returns:
            A boolean indicating whether the queried particle-type is supported.
        """
        return isinstance(particle_type, FermionicOp)

    def map(self, second_q_op: ParticleOp) -> PauliSumOp:
        """Maps a `ParticleOp` to a `PauliSumOp` using the Parity fermion-to-qubit
        mapping.

        Args:
            second_q_op: the `ParticleOp` to be mapped.

        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.

        Raises:
            QiskitNatureError: FermionicOp has a invalid label.
            TypeError: Type of second_q_op is not FermionicOp.
       """
        if not isinstance(second_q_op, FermionicOp):
            raise TypeError(
                f"Parity mapper only maps from FermionicOp, not {type(second_q_op)}"
            )

        # number of modes/sites for the Parity transform (= number of fermionc modes)
        nmodes = second_q_op.register_length

        pauli_table = []
        for i in range(nmodes):
            a_z: Union[List[int], np.ndarray] = [0] * (i - 1) + [1] if i > 0 else []
            a_x: Union[List[int], np.ndarray] = [0] * (i - 1) + [0] if i > 0 else []
            b_z: Union[List[int], np.ndarray] = [0] * (i - 1) + [0] if i > 0 else []
            b_x: Union[List[int], np.ndarray] = [0] * (i - 1) + [0] if i > 0 else []
            a_z = np.asarray(a_z + [0] + [0] * (nmodes - i - 1), dtype=bool)
            a_x = np.asarray(a_x + [1] + [1] * (nmodes - i - 1), dtype=bool)
            b_z = np.asarray(b_z + [1] + [0] * (nmodes - i - 1), dtype=bool)
            b_x = np.asarray(b_x + [1] + [1] * (nmodes - i - 1), dtype=bool)
            pauli_table.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))

        return QubitMapper.mode_based_mapping(second_q_op, pauli_table)
