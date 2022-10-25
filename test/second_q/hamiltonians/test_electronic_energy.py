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

"""Test ElectronicEnergy Property"""

import json
import unittest
from test.second_q.properties.property_test import PropertyTest
from typing import cast

import numpy as np

import qiskit_nature.optionals as _optionals
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals, PolynomialTensor


@unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
class TestElectronicEnergy(PropertyTest):
    """Test ElectronicEnergy Property"""

    def setUp(self):
        """Setup."""
        super().setUp()
        driver = PySCFDriver()
        self.prop = cast(ElectronicEnergy, driver.run().hamiltonian)

    def test_second_q_op(self):
        """Test second_q_op."""
        op = self.prop.second_q_op()
        with open(
            self.get_resource_path("electronic_energy_op.json", "second_q/hamiltonians/resources"),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)
        for (key1, val1), (key2, val2) in zip(sorted(op.items()), sorted(expected.items())):
            self.assertEqual(key1, key2)
            self.assertTrue(np.isclose(np.abs(val1), np.abs(val2)))

    def test_fock(self):
        """Test fock."""
        density = ElectronicIntegrals(alpha=PolynomialTensor({"+-": 0.5 * np.eye(2)}))
        fock_op = self.prop.fock(density)

        expected = np.asarray([[-0.34436786423711596, 0.0], [0.0, 0.4515069814257469]])
        self.assertTrue(np.allclose(fock_op.alpha["+-"], expected))
        self.assertNotIn("++--", fock_op.alpha)
        self.assertNotIn("++--", fock_op.beta)
        self.assertNotIn("++--", fock_op.beta_alpha)

    def test_from_raw_integrals(self):
        """Test from_raw_integrals utility method."""
        one_body_a = np.random.random((2, 2))
        one_body_b = np.random.random((2, 2))
        two_body_aa = np.random.random((2, 2, 2, 2))
        two_body_bb = np.random.random((2, 2, 2, 2))
        two_body_ba = np.random.random((2, 2, 2, 2))

        with self.subTest("alpha only"):
            prop = ElectronicEnergy.from_raw_integrals(
                one_body_a, two_body_aa, auto_index_order=False
            )
            self.assertTrue(np.allclose(prop.electronic_integrals.alpha["+-"], one_body_a))
            self.assertTrue(np.allclose(prop.electronic_integrals.alpha["++--"], two_body_aa))

        with self.subTest("alpha and beta"):
            prop = ElectronicEnergy.from_raw_integrals(
                one_body_a,
                two_body_aa,
                h1_b=one_body_b,
                h2_bb=two_body_bb,
                h2_ba=two_body_ba,
                auto_index_order=False,
            )
            self.assertTrue(np.allclose(prop.electronic_integrals.alpha["+-"], one_body_a))
            self.assertTrue(np.allclose(prop.electronic_integrals.beta["+-"], one_body_b))
            self.assertTrue(np.allclose(prop.electronic_integrals.alpha["++--"], two_body_aa))
            self.assertTrue(np.allclose(prop.electronic_integrals.beta["++--"], two_body_bb))
            self.assertTrue(np.allclose(prop.electronic_integrals.beta_alpha["++--"], two_body_ba))


if __name__ == "__main__":
    unittest.main()
