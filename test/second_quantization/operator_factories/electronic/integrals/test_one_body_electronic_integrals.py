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

"""Test OneBodyElectronicIntegrals."""

import tempfile
from test.properties.property_test import PropertyTest

import h5py
import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_quantization.operator_factories.electronic.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.second_quantization.operator_factories.electronic.integrals import (
    OneBodyElectronicIntegrals,
)


class TestOneBodyElectronicIntegrals(PropertyTest):
    """Test OneBodyElectronicIntegrals."""

    def test_init(self):
        """Test construction."""
        random = np.random.rand(2, 2)

        with self.subTest("Normal"):
            OneBodyElectronicIntegrals(ElectronicBasis.MO, (random, None))

        with self.subTest("Alpha and beta"):
            OneBodyElectronicIntegrals(ElectronicBasis.MO, (random, random))

        with self.subTest("Spin"):
            OneBodyElectronicIntegrals(ElectronicBasis.SO, random)

        with self.subTest("Mismatching basis and number of matrices"):
            with self.assertRaises(TypeError):
                OneBodyElectronicIntegrals(ElectronicBasis.MO, random)

        with self.subTest("Mismatching basis and number of matrices 2"):
            with self.assertRaises(TypeError):
                OneBodyElectronicIntegrals(ElectronicBasis.SO, (random, random))

        with self.subTest("Missing alpha"):
            with self.assertRaises(TypeError):
                OneBodyElectronicIntegrals(ElectronicBasis.MO, (None, random))

    def test_transform_basis(self):
        """Test transform_basis"""
        mat_a = np.arange(1, 5).reshape((2, 2))
        mat_b = np.arange(-4, 0).reshape((2, 2))

        transform = ElectronicBasisTransform(ElectronicBasis.AO, ElectronicBasis.MO, 2 * np.eye(2))

        with self.subTest("Pure Alpha"):
            ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (mat_a, None))
            ints_mo = ints_ao.transform_basis(transform)
            self.assertTrue(np.allclose(ints_mo._matrices[0], 4 * mat_a))
            self.assertIsNone(ints_mo._matrices[1])

        with self.subTest("Alpha and Beta"):
            ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (mat_a, mat_b))
            ints_mo = ints_ao.transform_basis(transform)
            self.assertTrue(np.allclose(ints_mo._matrices[0], 4 * mat_a))
            self.assertTrue(np.allclose(ints_mo._matrices[1], 4 * mat_b))

        with self.subTest("Beta custom coeff with only alpha"):
            transform_beta = ElectronicBasisTransform(
                ElectronicBasis.AO, ElectronicBasis.MO, 2 * np.eye(2), 3 * np.eye(2)
            )
            ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (mat_a, None))
            ints_mo = ints_ao.transform_basis(transform_beta)
            self.assertTrue(np.allclose(ints_mo._matrices[0], 4 * mat_a))
            self.assertTrue(np.allclose(ints_mo._matrices[1], 9 * mat_a))

        with self.subTest("Beta custom coeff"):
            transform_beta = ElectronicBasisTransform(
                ElectronicBasis.AO, ElectronicBasis.MO, 2 * np.eye(2), 3 * np.eye(2)
            )
            ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (mat_a, mat_b))
            ints_mo = ints_ao.transform_basis(transform_beta)
            self.assertTrue(np.allclose(ints_mo._matrices[0], 4 * mat_a))
            self.assertTrue(np.allclose(ints_mo._matrices[1], 9 * mat_b))

        with self.subTest("Final basis match"):
            ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, None))
            ints_mo = ints_ao.transform_basis(transform)
            self.assertEqual(ints_ao, ints_mo)

        with self.subTest("Inital basis mismatch"):
            with self.assertRaises(QiskitNatureError):
                ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.SO, mat_a)
                ints_ao.transform_basis(transform)

    def test_to_spin(self):
        """Test to_spin"""
        mat_a = np.arange(1, 5).reshape((2, 2))
        mat_b = np.arange(-4, 0).reshape((2, 2))

        with self.subTest("Only alpha"):
            ints = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, None))
            mat_so = ints.to_spin()
            self.assertTrue(
                np.allclose(
                    mat_so, np.asarray([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4]])
                )
            )

        with self.subTest("Alpha and beta"):
            ints = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, mat_b))
            mat_so = ints.to_spin()
            self.assertTrue(
                np.allclose(
                    mat_so, np.asarray([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, -4, -3], [0, 0, -2, -1]])
                )
            )

    def test_to_second_q_op(self):
        """Test to_second_q_op"""
        mat_a = np.arange(1, 5).reshape((2, 2))
        mat_b = np.arange(-4, 0).reshape((2, 2))

        with self.subTest("Only alpha"):
            ints = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, None))
            op = ints.to_second_q_op()
            for (real_label, real_coeff), (exp_label, exp_coeff) in zip(
                op.to_list(),
                [
                    ("+_0 -_0", 1),
                    ("+_0 -_1", 2),
                    ("+_1 -_0", 3),
                    ("+_1 -_1", 4),
                    ("+_2 -_2", 1),
                    ("+_2 -_3", 2),
                    ("+_3 -_2", 3),
                    ("+_3 -_3", 4),
                ],
            ):
                self.assertEqual(real_label, exp_label)
                self.assertTrue(np.isclose(real_coeff, exp_coeff))

        with self.subTest("Alpha and beta"):
            ints = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, mat_b))
            op = ints.to_second_q_op()
            for (real_label, real_coeff), (exp_label, exp_coeff) in zip(
                op.to_list(),
                [
                    ("+_0 -_0", 1),
                    ("+_0 -_1", 2),
                    ("+_1 -_0", 3),
                    ("+_1 -_1", 4),
                    ("+_2 -_2", -4),
                    ("+_2 -_3", -3),
                    ("+_3 -_2", -2),
                    ("+_3 -_3", -1),
                ],
            ):
                self.assertEqual(real_label, exp_label)
                self.assertTrue(np.isclose(real_coeff, exp_coeff))

    def test_add(self):
        """Test addition."""
        mat_a = np.arange(1, 5).reshape((2, 2))
        mat_b = np.arange(-4, 0).reshape((2, 2))

        ints_a = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, None))
        ints_b = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_b, None))

        ints_sum = ints_a + ints_b

        self.assertTrue(isinstance(ints_sum, OneBodyElectronicIntegrals))
        self.assertTrue(np.allclose(ints_sum._matrices[0], mat_a + mat_b))

    def test_mul(self):
        """Test multiplication."""
        mat_a = np.arange(1, 5).reshape((2, 2))

        ints_a = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, None))

        ints_mul = 2.0 * ints_a

        self.assertTrue(isinstance(ints_mul, OneBodyElectronicIntegrals))
        self.assertTrue(np.allclose(ints_mul._matrices[0], 2.0 * mat_a))

    def test_compose(self):
        """Test composition."""
        mat_a = np.arange(1, 5).reshape((2, 2))
        mat_b = np.arange(-4, 0).reshape((2, 2))

        ints_a = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, None))
        ints_b = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_b, None))

        composition = ints_a.compose(ints_b)

        # The factor 2.0 arises from the fact that mat_a and mat_b also get populated into the None
        # fields.
        expected = 2.0 * np.einsum("ij,ji", mat_a, mat_b)

        self.assertTrue(isinstance(composition, complex))
        self.assertAlmostEqual(composition, expected)

    def test_to_hdf5(self):
        """Test to_hdf5."""
        random = np.random.rand(2, 2)

        ints = OneBodyElectronicIntegrals(ElectronicBasis.MO, (random, random))

        with tempfile.TemporaryFile() as tmp_file:
            with h5py.File(tmp_file, "w") as file:
                ints.to_hdf5(file)

    def test_from_hdf5(self):
        """Test from_hdf5."""
        random = np.random.rand(2, 2)

        ints = OneBodyElectronicIntegrals(ElectronicBasis.MO, (random, random))

        with tempfile.TemporaryFile() as tmp_file:
            with h5py.File(tmp_file, "w") as file:
                ints.to_hdf5(file)

            with h5py.File(tmp_file, "r") as file:
                new_ints = OneBodyElectronicIntegrals.from_hdf5(file["OneBodyElectronicIntegrals"])

                self.assertEqual(ints, new_ints)
