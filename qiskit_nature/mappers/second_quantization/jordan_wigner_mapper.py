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

"""The Jordan-Wigner Mapper. """

import numpy as np
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli

from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.operators.second_quantization.particle_op import ParticleOp

from .qubit_mapper import QubitMapper


class JordanWignerMapper(QubitMapper):
    """The Jordan-Wigner fermion-to-qubit mapping. """

    def supports_particle_type(self, particle_type: ParticleOp) -> bool:
        """Returns whether the queried particle-type operator is supported by this mapping.

        Args:
            particle_type: the particle-type to query support for.

        Returns:
            A boolean indicating whether the queried particle-type is supported.
        """
        return isinstance(particle_type, FermionicOp)

    def map(self, second_q_op: ParticleOp) -> PauliSumOp:
        """Maps a `SecondQuantizedOp` to a `PauliSumOp` using the Jordan-Wigner
        fermion-to-qubit mapping.

        Args:
            second_q_op: the `SecondQuantizedOp` to be mapped.
        Returns:
            The `PauliSumOp` corresponding to the problem-Hamiltonian in the qubit space.
        Raises:
            QiskitNatureError: FermionicOp has a invalid label.
            TypeError: Type of second_q_op is not FermionicOp.
        """
        if not isinstance(second_q_op, FermionicOp):
            raise TypeError(
                f"Jordan-Wigner mapper only maps from FermionicOp, not {type(second_q_op)}"
            )

        # number of modes/sites for the Jordan-Wigner transform (= number of fermionc modes)
        nmodes = second_q_op.register_length

        pauli_table = []
        for i in range(nmodes):
            a_z = np.asarray([1] * i + [0] + [0] * (nmodes - i - 1), dtype=bool)
            a_x = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
            b_z = np.asarray([1] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
            b_x = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
            # c_z = np.asarray([0] * i + [1] + [0] * (nmodes - i - 1), dtype=bool)
            # c_x = np.asarray([0] * nmodes, dtype=bool)
            pauli_table.append((Pauli((a_z, a_x)), Pauli((b_z, b_x))))
            # TODO add Pauli 3-tuple to lookup table

        return QubitMapping.mode_based_mapping(second_q_op, pauli_table)
