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

"""Test AngularMomentum Property"""

import json
import numpy as np

from test import QiskitNatureTestCase

from qiskit_nature.drivers.second_quantization import QMolecule
from qiskit_nature.properties.electronic_structure import AngularMomentum


class TestAngularMomentum(QiskitNatureTestCase):
    def setUp(self):
        """Setup."""
        super().setUp()
        qmol = QMolecule()
        qmol.num_molecular_orbitals = 4
        self.prop = AngularMomentum.from_driver_result(qmol)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = self.prop.second_q_ops()
        assert len(ops) == 1
        with open(
            self.get_resource_path(
                "angular_momentum_op.json", "properties/electronic_structure/resources"
            ),
            "r",
        ) as f:
            expected = json.load(f)
        for op, expected_op in zip(ops[0].to_list(), expected):
            assert op[0] == expected_op[0]
            assert np.isclose(op[1], expected_op[1])
