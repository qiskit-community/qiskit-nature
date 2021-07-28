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

"""Test ElectronicEnergy Property"""

import json
from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy
from qiskit_nature.properties.second_quantization.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
)


class TestElectronicEnergy(QiskitNatureTestCase):
    """Test ElectronicEnergy Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "test_driver_hdf5.hdf5", "drivers/second_quantization/hdf5d"
            )
        )
        qmol = driver.run()
        self.prop = ElectronicEnergy.from_legacy_driver_result(qmol)

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = self.prop.second_q_ops()
        self.assertEqual(len(ops), 1)
        with open(
            self.get_resource_path(
                "electronic_energy_op.json", "properties/second_quantization/electronic/resources"
            ),
            "r",
        ) as file:
            expected = json.load(file)
        for op, expected_op in zip(ops[0].to_list(), expected):
            self.assertEqual(op[0], expected_op[0])
            self.assertTrue(np.isclose(op[1], expected_op[1]))

    def test_integral_operator(self):
        """Test integral_operator."""
        # duplicate MO integrals into AO basis for this test
        trafo = ElectronicBasisTransform(ElectronicBasis.MO, ElectronicBasis.AO, np.eye(2))
        self.prop.transform_basis(trafo)

        density = OneBodyElectronicIntegrals(ElectronicBasis.AO, (0.5 * np.eye(2), None))
        matrix_op = self.prop.integral_operator(density)

        expected = np.asarray([[-0.34436786423711596, 0.0], [0.0, 0.4515069814257469]])
        self.assertTrue(np.allclose(matrix_op._matrices[0], expected))
