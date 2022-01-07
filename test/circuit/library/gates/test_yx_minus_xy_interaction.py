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
from ddt import data, ddt, unpack

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_nature.circuit.library import YXMinusXYInteractionGate


@ddt
class TestYXMinusXYInteractionGate(QiskitNatureTestCase):
    """Tests for YXMinusXYInteractionGate gate"""

    @unpack
    @data(
        (0, np.eye(4)),
        (
            np.pi / 4,
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                    [0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
                    [0, 0, 0, 1],
                ]
            ),
        ),
        (np.pi / 2, np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])),
        (np.pi, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])),
        (2 * np.pi, np.eye(4)),
    )
    def test_matrix(self, theta: float, expected: np.ndarray):
        """Test matrix."""
        gate = YXMinusXYInteractionGate(theta)
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
