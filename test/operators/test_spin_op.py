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

"""Test for SpinOp"""

import unittest

from test import QiskitNatureTestCase

from qiskit_nature.operators import SpinOp


class TestSpinOp(QiskitNatureTestCase):
    """FermionicOp tests."""

    def test_init(self):
        """Test __init__"""
        heisenberg = SpinOp([
            ([1, 1], [0, 0], [0, 0], -1),
            ([0, 0], [1, 1], [0, 0], -1),
            ([0, 0], [0, 0], [1, 1], -1),
            ([0, 0], [0, 0], [1, 0], -0.3),
            ([0, 0], [0, 0], [0, 1], -0.3),
            ], spin=1)
        print(heisenberg)


if __name__ == "__main__":
    unittest.main()
