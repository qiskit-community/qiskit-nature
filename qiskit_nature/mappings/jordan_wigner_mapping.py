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

"""The Jordan-Wigner Mapping interface."""

import numpy as np
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli, SparsePauliOp

from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.operators.second_quantization.particle_op import ParticleOp

from .qubit_mapping import QubitMapping


class JordanWignerMapping(QubitMapping):
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

        # 0. Some utilities
        def times_creation_op(op, position, pauli_table):
            # The creation operator is given by 0.5*(X + 1j*Y)
            real_part = SparsePauliOp(pauli_table[position][0], coeffs=[0.5])
            imag_part = SparsePauliOp(pauli_table[position][1], coeffs=[0.5j])

            # We must multiply from the left due to the right-to-left execution order of operators.
            prod = (real_part + imag_part) * op
            return prod

        def times_annihilation_op(op, position, pauli_table):
            # The annihilation operator is given by 0.5*(X - 1j*Y)
            real_part = SparsePauliOp(pauli_table[position][0], coeffs=[0.5])
            imag_part = SparsePauliOp(pauli_table[position][1], coeffs=[-0.5j])

            # We must multiply from the left due to the right-to-left execution order of operators.
            prod = (real_part + imag_part) * op
            return prod

        # 1. Initialize an operator list with the identity scaled by the `self.coeff`
        all_false = np.asarray([False] * nmodes, dtype=np.bool)

        ret_op_list = []

        for label, coeff in second_q_op.to_list():

            ret_op = SparsePauliOp(Pauli((all_false, all_false)), coeffs=[coeff])

            # Go through the label and replace the fermion operators by their qubit-equivalent, then
            # save the respective Pauli string in the pauli_str list.
            for position, char in enumerate(label):
                if char == "+":
                    ret_op = times_creation_op(ret_op, position, pauli_table)
                elif char == "-":
                    ret_op = times_annihilation_op(ret_op, position, pauli_table)
                elif char == "N":
                    # The occupation number operator N is given by `+-`.
                    ret_op = times_creation_op(ret_op, position, pauli_table)
                    ret_op = times_annihilation_op(ret_op, position, pauli_table)
                elif char == "E":
                    # The `emptiness number` operator E is given by `-+` = (I - N).
                    ret_op = times_annihilation_op(ret_op, position, pauli_table)
                    ret_op = times_creation_op(ret_op, position, pauli_table)
                elif char == "I":
                    continue

                # catch any disallowed labels
                else:
                    raise QiskitNatureError(
                        f"BaseFermionOperator label included '{char}'. "
                        "Allowed characters: I, N, E, +, -"
                    )
            ret_op_list.append(ret_op)

        zero_op = SparsePauliOp(Pauli((all_false, all_false)), coeffs=[0])
        return PauliSumOp(sum(ret_op_list, zero_op)).reduce()
