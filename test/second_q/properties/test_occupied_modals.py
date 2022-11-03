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

"""Test OccupiedModals Property"""

from test.second_q.properties.property_test import PropertyTest

from qiskit_nature.second_q.operators import VibrationalOp
from qiskit_nature.second_q.properties import OccupiedModals


class TestOccupiedModals(PropertyTest):
    """Test OccupiedModals Property"""

    def setUp(self):
        """Setup basis."""
        super().setUp()
        self.num_modals = [2, 3, 4]
        self.prop = OccupiedModals(self.num_modals)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = self.prop.second_q_ops()
        expected = [
            VibrationalOp({"+_0_0 -_0_0": 1.0, "+_0_1 -_0_1": 1.0}, num_modals=self.num_modals),
            VibrationalOp(
                {"+_1_0 -_1_0": 1.0, "+_1_1 -_1_1": 1.0, "+_1_2 -_1_2": 1.0},
                num_modals=self.num_modals,
            ),
            VibrationalOp(
                {"+_2_0 -_2_0": 1.0, "+_2_1 -_2_1": 1.0, "+_2_2 -_2_2": 1.0, "+_2_3 -_2_3": 1.0},
                num_modals=self.num_modals,
            ),
        ]

        for op, expected_op_list in zip(ops.values(), expected):
            self.assertEqual(op, expected_op_list)
