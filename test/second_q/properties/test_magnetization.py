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

"""Test Magnetization Property"""

from test.second_q.properties.property_test import PropertyTest

from qiskit_nature.second_q.properties import Magnetization


class TestMagnetization(PropertyTest):
    """Test Magnetization Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        num_spatial_orbitals = 4
        self.prop = Magnetization(num_spatial_orbitals)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = self.prop.second_q_ops()["Magnetization"]
        expected = {
            "+_0 -_0": 0.5,
            "+_1 -_1": 0.5,
            "+_2 -_2": 0.5,
            "+_3 -_3": 0.5,
            "+_4 -_4": -0.5,
            "+_5 -_5": -0.5,
            "+_6 -_6": -0.5,
            "+_7 -_7": -0.5,
        }
        self.assertEqual(dict(ops.items()), expected)
