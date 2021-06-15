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
from qiskit.opflow import I, Z, PauliSumOp

from problems.sampling.protein_folding.qubit_utils.qubit_fixing import _fix_qubits
from test import QiskitNatureTestCase


class TestQubitFixing(QiskitNatureTestCase):
    """Tests DistanceCalculator."""

    def test_fix_qubits_small(self):
        operator = (I ^ I ^ Z ^ Z) + (Z ^ I ^ I ^ I)
        fixed = _fix_qubits(operator)
        expected = PauliSumOp.from_list([("IIII", 0)])
        assert fixed == expected

    def test_fix_qubits_small_2(self):
        operator = (Z^Z) + (I^I)
        fixed = _fix_qubits(operator)
        expected = PauliSumOp.from_list([("II", 0)])
        assert fixed == expected

    def test_fix_qubits_large(self):
        operator = (Z ^ I ^ I ^ I ^ I ^ Z ^ Z) + (I ^ I ^ I ^ Z ^ I ^ I ^ I)
        fixed = _fix_qubits(operator)
        expected = (I ^ I ^ I ^ I ^ I ^ I ^ I) - (Z ^ I ^ I ^ I ^ I ^ I ^ I)
        assert fixed == expected

    def test_fix_qubits_pauli_op(self):
        operator = Z ^ I ^ I ^ I ^ I ^ Z ^ Z
        fixed = _fix_qubits(operator)
        expected = Z ^ I ^ I ^ I ^ I ^ I ^ I
        assert fixed == expected
