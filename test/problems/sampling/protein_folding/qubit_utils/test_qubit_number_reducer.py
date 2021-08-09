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
"""Tests QubitNumberReducer."""
from test import QiskitNatureTestCase
from qiskit.opflow import I, Z
from qiskit_nature.problems.sampling.protein_folding.qubit_utils.qubit_number_reducer import (
    _find_unused_qubits,
    _remove_unused_qubits,
)


class TestQubitNumberReducer(QiskitNatureTestCase):
    """Tests ContactQubitsBuilder."""

    def test_find_unused_qubits(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = (I ^ I ^ I) + (Z ^ I ^ Z)
        unused = _find_unused_qubits(operator)
        expected = [1]
        self.assertEqual(unused, expected)

    def test_find_unused_qubits_2(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = (I ^ I ^ I) + (Z ^ Z ^ Z)
        unused = _find_unused_qubits(operator)
        expected = []
        self.assertEqual(unused, expected)

    def test_find_unused_qubits_3(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = (Z ^ I ^ Z) + (I ^ Z ^ I)
        unused = _find_unused_qubits(operator)
        expected = []
        self.assertEqual(unused, expected)

    def test_find_unused_qubits_4(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = (I ^ I ^ I ^ I ^ I) + (Z ^ I ^ I ^ I ^ I) + (I ^ I ^ I ^ I ^ Z)
        unused = _find_unused_qubits(operator)
        expected = [1, 2, 3]
        self.assertEqual(unused, expected)

    def test_find_unused_qubits_5(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = (I ^ I ^ I ^ Z ^ I) + (I ^ I ^ Z ^ I ^ I)
        unused = _find_unused_qubits(operator)
        expected = [0, 3, 4]
        self.assertEqual(unused, expected)

    def test_find_unused_qubits_pauli_op(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = I ^ I ^ I ^ Z ^ I
        unused = _find_unused_qubits(operator)
        expected = [0, 2, 3, 4]
        self.assertEqual(unused, expected)

    def test_remove_unused_qubits(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = (I ^ I ^ I ^ I ^ I) + (Z ^ I ^ I ^ I ^ I) + (I ^ I ^ I ^ I ^ Z)

        reduced, unused = _remove_unused_qubits(operator)
        expected_reduced = 1.0 * (I ^ I) + 1.0 * (Z ^ I) + 1.0 * (I ^ Z)
        expected_unused = [1, 2, 3]
        self.assertEqual(reduced, expected_reduced)
        self.assertEqual(unused, expected_unused)

    def test_remove_unused_qubits_2(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = (I ^ I ^ I ^ Z ^ I) + (I ^ I ^ Z ^ I ^ I)
        reduced, unused = _remove_unused_qubits(operator)
        expected_reduced = 1.0 * (I ^ Z) + 1.0 * (Z ^ I)
        expected_unused = [0, 3, 4]
        self.assertEqual(reduced, expected_reduced)
        self.assertEqual(unused, expected_unused)

    def test_remove_unused_qubits_keep_all(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = (Z ^ I ^ Z) + (I ^ Z ^ I)
        reduced, unused = _remove_unused_qubits(operator)
        expected_reduced = 1.0 * (I ^ Z ^ I) + 1.0 * (Z ^ I ^ Z)
        expected_unused = []
        self.assertEqual(reduced, expected_reduced)
        self.assertEqual(unused, expected_unused)

    def test_remove_unused_qubits_pauli_op(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = I ^ I ^ I ^ Z ^ I
        reduced, unused = _remove_unused_qubits(operator)
        expected_reduced = 1.0 * (Z)
        expected_unused = [0, 2, 3, 4]
        self.assertEqual(reduced, expected_reduced)
        self.assertEqual(unused, expected_unused)

    def test_remove_unused_qubits_pauli_op_2(self):
        """
        Tests that unused qubits are found correctly.
        """
        operator = I ^ Z ^ I ^ Z ^ I
        reduced, unused = _remove_unused_qubits(operator)
        expected_reduced = 1.0 * (Z ^ Z)
        expected_unused = [0, 2, 4]
        self.assertEqual(reduced, expected_reduced)
        self.assertEqual(unused, expected_unused)
