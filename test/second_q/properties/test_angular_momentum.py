# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test AngularMomentum Property"""

import unittest
import json
from test.second_q.properties.property_test import PropertyTest

from qiskit_nature.second_q.properties import AngularMomentum
from qiskit_nature.second_q.operators import FermionicOp


class TestAngularMomentum(PropertyTest):
    """Test AngularMomentum Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        num_spatial_orbitals = 4
        self.prop = AngularMomentum(num_spatial_orbitals)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        op = self.prop.second_q_ops()["AngularMomentum"]
        with open(
            self.get_resource_path("angular_momentum_op.json", "second_q/properties/resources"),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)
            expected_op = FermionicOp(expected, num_spin_orbitals=8).simplify()
        self.assertEqual(op, expected_op)


if __name__ == "__main__":
    unittest.main()
