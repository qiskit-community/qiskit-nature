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

"""Test for commutators"""

from __future__ import annotations

import unittest
from test import QiskitNatureTestCase
from ddt import ddt, data, unpack

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


if __name__ == "__main__":
    unittest.main()
