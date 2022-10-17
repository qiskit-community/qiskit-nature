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

"""Test ParticleNumber Property"""

from test.second_q.properties.property_test import PropertyTest

from qiskit_nature.second_q.properties import ParticleNumber


class TestParticleNumber(PropertyTest):
    """Test ParticleNumber Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        num_spatial_orbitals = 4
        self.prop = ParticleNumber(num_spatial_orbitals)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = self.prop.second_q_ops()["ParticleNumber"]
        expected = {
            "+_0 -_0": 1.0,
            "+_1 -_1": 1.0,
            "+_2 -_2": 1.0,
            "+_3 -_3": 1.0,
            "+_4 -_4": 1.0,
            "+_5 -_5": 1.0,
            "+_6 -_6": 1.0,
            "+_7 -_7": 1.0,
        }
        self.assertEqual(dict(ops.items()), expected)
