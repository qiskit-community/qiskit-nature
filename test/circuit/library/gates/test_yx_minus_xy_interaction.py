# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test YXMinusXYInteractionGate."""

from test import QiskitNatureTestCase
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_nature.circuit.library import YXMinusXYInteractionGate


class TestYXMinusXYInteractionGate(QiskitNatureTestCase):
    """Tests for YXMinusXYInteractionGate gate"""

    def test_matrix(self):
        """Test matrix."""
        gate = YXMinusXYInteractionGate(0)
        expected = np.eye(4)
        np.testing.assert_allclose(gate.to_matrix(), expected, atol=1e-7)

        gate = YXMinusXYInteractionGate(np.pi / 4)
        a = np.sqrt(2) / 2
        expected = np.array([[1, 0, 0, 0], [0, a, a, 0], [0, -a, a, 0], [0, 0, 0, 1]])
        np.testing.assert_allclose(gate.to_matrix(), expected, atol=1e-7)

        gate = YXMinusXYInteractionGate(np.pi / 2)
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_allclose(gate.to_matrix(), expected, atol=1e-7)

        gate = YXMinusXYInteractionGate(np.pi)
        expected = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        np.testing.assert_allclose(gate.to_matrix(), expected, atol=1e-7)

        gate = YXMinusXYInteractionGate(2 * np.pi)
        expected = np.eye(4)
        np.testing.assert_allclose(gate.to_matrix(), expected, atol=1e-7)

    def test_inverse(self):
        """Test inverse."""
        theta = np.random.uniform(-10, 10)
        gate = YXMinusXYInteractionGate(theta)
        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1])
        circuit.append(gate.inverse(), [0, 1])
        assert Operator(circuit).equiv(np.eye(4), atol=1e-7)

    def test_decompose(self):
        """Test decomposition."""
        theta = np.random.uniform(-10, 10)
        gate = YXMinusXYInteractionGate(theta)
        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1])
        decomposed_circuit = circuit.decompose()
        assert len(decomposed_circuit) > len(circuit)
        assert Operator(circuit).equiv(Operator(decomposed_circuit), atol=1e-7)
