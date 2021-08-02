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

"""Test ParticleNumber Property"""

from test import QiskitNatureTestCase

from qiskit_nature.drivers.second_quantization import QMolecule
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber


class TestParticleNumber(QiskitNatureTestCase):
    """Test ParticleNumber Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        qmol = QMolecule()
        qmol.num_molecular_orbitals = 4
        qmol.num_alpha = 2
        qmol.num_beta = 2
        self.prop = ParticleNumber.from_legacy_driver_result(qmol)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = self.prop.second_q_ops()
        self.assertEqual(len(ops), 1)
        expected = [
            "NIIIIIII",
            "INIIIIII",
            "IINIIIII",
            "IIINIIII",
            "IIIINIII",
            "IIIIINII",
            "IIIIIINI",
            "IIIIIIIN",
        ]
        self.assertEqual([l for l, _ in ops[0].to_list()], expected)
