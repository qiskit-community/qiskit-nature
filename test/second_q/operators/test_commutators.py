# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for commutators"""

from __future__ import annotations

import unittest
from test import QiskitNatureTestCase
from ddt import ddt, data, unpack

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators.commutators import (
    commutator,
    anti_commutator,
    double_commutator,
)

op1 = FermionicOp({"+_0 -_0": 1}, num_spin_orbitals=1)
op2 = FermionicOp({"-_0 +_0": 2}, num_spin_orbitals=1)
op3 = FermionicOp({"+_0 -_0": 1, "-_0 +_0": 2 + 0.5j}, num_spin_orbitals=1)
op4 = FermionicOp({"+_0": 1}, num_spin_orbitals=1)
op5 = FermionicOp({"-_0": 1}, num_spin_orbitals=1)

# ParityMapper
op1_pauli = SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)])
op2_pauli = SparsePauliOp.from_list([("I", 1.0), ("Z", 1.0)])
op3_pauli = SparsePauliOp.from_list([("I", 1.5 + 0.25j), ("Z", 0.5 + 0.25j)])
op4_pauli = SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)])
op5_pauli = SparsePauliOp.from_list([("X", 0.5), ("Y", 0.5j)])


@ddt
class TestCommutators(QiskitNatureTestCase):
    """Commutators tests."""

    @unpack
    @data(
        (op1, op2, {}),
        (op4, op5, {"+_0 -_0": (1 + 0j), "-_0 +_0": (-1 + 0j)}),
    )
    def test_commutator(self, op_a: FermionicOp, op_b: FermionicOp, expected: dict):
        """Test commutator method"""
        self.assertEqual(commutator(op_a, op_b), FermionicOp(expected, num_spin_orbitals=1))

    @unpack
    @data(
        (op1, op2, {}),
        (op1, op3, {"+_0 -_0": (2 + 0j)}),
    )
    def test_anti_commutator(self, op_a: FermionicOp, op_b: FermionicOp, expected: dict):
        """Test anti commutator method"""
        self.assertEqual(anti_commutator(op_a, op_b), FermionicOp(expected, num_spin_orbitals=1))

    @unpack
    @data(
        (op1, op2, op3, False, {}),
        (op1, op4, op3, False, {"+_0": (1 + 0.5j)}),
        (op1, op4, op3, True, {"+_0": (2 + 0.5j)}),
    )
    def test_double_commutator(
        self,
        op_a: FermionicOp,
        op_b: FermionicOp,
        op_c: FermionicOp,
        sign: bool,
        expected: dict,
    ):
        """Test double commutator method"""
        self.assertEqual(
            double_commutator(op_a, op_b, op_c, sign), FermionicOp(expected, num_spin_orbitals=1)
        )

    @unpack
    @data(
        (op1_pauli, op2_pauli, [("I", 0.0)]),
        (op4_pauli, op5_pauli, [("Z", -1)]),
    )
    def test_commutator_pauli(self, op_a: SparsePauliOp, op_b: SparsePauliOp, expected: list):
        """Test commutator method"""
        self.assertEqual(commutator(op_a, op_b), SparsePauliOp.from_list(expected))

    @unpack
    @data(
        (op1_pauli, op2_pauli, [("I", 0.0)]),
        (op1_pauli, op3_pauli, [("I", 1), ("Z", -1)]),
    )
    def test_anti_commutator_pauli(self, op_a: SparsePauliOp, op_b: SparsePauliOp, expected: list):
        """Test anti commutator method"""
        self.assertEqual(anti_commutator(op_a, op_b), SparsePauliOp.from_list(expected))

    @unpack
    @data(
        (op1_pauli, op2_pauli, op3_pauli, False, [("I", 0.0)]),
        (op1_pauli, op4_pauli, op3_pauli, False, [("X", (0.5 + 0.25j)), ("Y", (-0.5j + 0.25))]),
        (op1_pauli, op4_pauli, op3_pauli, True, [("X", (1 + 0.25j)), ("Y", (-1j + 0.25))]),
    )
    def test_double_commutator_pauli(
        self,
        op_a: SparsePauliOp,
        op_b: SparsePauliOp,
        op_c: SparsePauliOp,
        sign: bool,
        expected: list,
    ):
        """Test double commutator method"""
        self.assertEqual(
            double_commutator(op_a, op_b, op_c, sign), SparsePauliOp.from_list(expected)
        )


if __name__ == "__main__":
    unittest.main()
