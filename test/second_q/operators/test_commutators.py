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

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators.commutators import (
    commutator,
    anti_commutator,
    double_commutator
)

op1 = FermionicOp({"+_0 -_0": 1}, register_length=1)
op2 = FermionicOp({"-_0 +_0": 2}, register_length=1)
op3 = FermionicOp({"+_0 -_0": 1, "-_0 +_0": 2}, register_length=1)

class TestCommutators(QiskitNatureTestCase):
    """Commutators tests."""

    def test_commutator(self):
        """Test commutator method"""
        commutator(op1, op2)

    def test_anti_commutator(self):
        """Test anti commutator method"""
        anti_commutator(op1, op2)

    def test_double_commutator(self):
        """Test double commutator method"""
        double_commutator(op1, op2, op3)


if __name__ == "__main__":
    unittest.main()
