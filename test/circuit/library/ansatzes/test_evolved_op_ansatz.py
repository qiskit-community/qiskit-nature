# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the evolved operator ansatz."""

from test import QiskitNatureTestCase
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import X, Y, Z, I, MatrixEvolution, Suzuki

from qiskit_nature.circuit.library.ansatzes import EvolvedOperatorAnsatz


@ddt
class TestEvolvedOperatorAnsatz(QiskitNatureTestCase):
    """Test the evolved operator ansatz."""

    def test_evolved_op_ansatz(self):
        """Test the default evolution."""
        num_qubits = 3

        ops = [Z ^ num_qubits, Y ^ num_qubits, X ^ num_qubits]
        strings = ['z' * num_qubits, 'y' * num_qubits, 'x' * num_qubits]

        evo = EvolvedOperatorAnsatz(ops)
        evo._build()  # fixed by speedup parameter binds PR

        reference = QuantumCircuit(num_qubits)
        parameters = evo.parameters
        for string, time in zip(strings, parameters):
            reference.compose(evolve(string, time), inplace=True)

        self.assertEqual(evo, reference)

    def test_custom_evolution(self):
        """Test using another evolution than the default (e.g. matrix evo)."""

        op = X ^ I ^ Z
        matrix = op.to_matrix()
        evo = EvolvedOperatorAnsatz(op, evolution=MatrixEvolution())
        evo._build()

        parameters = evo.parameters
        reference = QuantumCircuit(3)
        reference.hamiltonian(matrix, parameters[0], [0, 1, 2])

        self.assertEqual(evo, reference)

    def test_changing_operators(self):
        """Test rebuilding after the operators changed."""

        ops = [X, Y, Z]
        evo = EvolvedOperatorAnsatz(ops)
        evo.operators = [X, Y]
        evo._build()

        parameters = evo.parameters
        reference = QuantumCircuit(1)
        reference.rx(2 * parameters[0], 0)
        reference.ry(2 * parameters[1], 0)

        print(evo, reference)
        self.assertEqual(evo, reference)


def evolve(pauli_string, time):
    num_qubits = len(pauli_string)
    forward = QuantumCircuit(num_qubits)
    for i, pauli in enumerate(pauli_string):
        if pauli == 'x':
            forward.h(i)
        elif pauli == 'y':
            forward.sdg(i)
            forward.h(i)

    for i in range(1, num_qubits):
        forward.cx(i, 0)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(forward, inplace=True)
    circuit.rz(2 * time, 0)
    circuit.compose(forward.inverse(), inplace=True)

    return circuit
