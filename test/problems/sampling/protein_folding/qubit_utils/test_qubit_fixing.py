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
"""Tests QubitFixing."""
from test import QiskitNatureTestCase
from qiskit.opflow import I, Z, PauliSumOp
from qiskit_nature.problems.sampling.protein_folding.qubit_utils.qubit_fixing import _fix_qubits


class TestQubitFixing(QiskitNatureTestCase):
    """Tests QubitFixing."""

    def test_fix_qubits_small(self):
        """Tests if qubits are fixed correctly for an operator on a small number of qubits."""
        operator = (I ^ I ^ Z ^ Z) + (Z ^ I ^ I ^ I)
        fixed = _fix_qubits(operator)
        expected = PauliSumOp.from_list([("IIII", 0)])
        self.assertEqual(fixed, expected)

    def test_fix_qubits_small_2(self):
        """Tests if qubits are fixed correctly for an operator on a small number of qubits."""
        operator = (Z ^ Z) + (I ^ I)
        fixed = _fix_qubits(operator)
        expected = PauliSumOp.from_list([("II", 0)])
        self.assertEqual(fixed, expected)

    def test_fix_qubits_large(self):
        """Tests if qubits are fixed correctly for an operator on a large number of qubits."""
        operator = (Z ^ I ^ I ^ I ^ I ^ Z ^ Z) + (I ^ I ^ I ^ Z ^ I ^ I ^ I)
        fixed = _fix_qubits(operator)
        expected = (I ^ I ^ I ^ I ^ I ^ I ^ I) - (Z ^ I ^ I ^ I ^ I ^ I ^ I)
        self.assertEqual(fixed, expected)

    def test_fix_qubits_pauli_op(self):
        """Tests if qubits are fixed correctly for an operator which is a PauliOp."""
        operator = Z ^ I ^ I ^ I ^ I ^ Z ^ Z
        fixed = _fix_qubits(operator)
        expected = Z ^ I ^ I ^ I ^ I ^ I ^ I
        self.assertEqual(fixed, expected)
