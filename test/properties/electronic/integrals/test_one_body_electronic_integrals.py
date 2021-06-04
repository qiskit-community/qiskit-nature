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

"""Test OneBodyElectronicIntegrals."""

from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.properties.electronic.bases import ElectronicBasis, ElectronicBasisTransform
from qiskit_nature.properties.electronic.integrals import OneBodyElectronicIntegrals


class TestOneBodyElectronicIntegrals(QiskitNatureTestCase):
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
            with self.assertRaises(AssertionError):
                OneBodyElectronicIntegrals(ElectronicBasis.MO, random)

        with self.subTest("Mismatching basis and number of matrices 2"):
            with self.assertRaises(AssertionError):
                OneBodyElectronicIntegrals(ElectronicBasis.SO, (random, random))

        with self.subTest("Missing alpha"):
            with self.assertRaises(AssertionError):
                OneBodyElectronicIntegrals(ElectronicBasis.MO, (None, random))

    def test_transform_basis(self):
        """Test transform_basis"""
        mat_a = np.arange(1, 5).reshape((2, 2))
        mat_b = np.arange(-4, 0).reshape((2, 2))

        transform = ElectronicBasisTransform(ElectronicBasis.AO, ElectronicBasis.MO, 2 * np.eye(2))

        with self.subTest("Pure Alpha"):
            ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (mat_a, None))
            ints_mo = ints_ao.transform_basis(transform)
            assert np.allclose(ints_mo._matrices[0], 4 * mat_a)
            assert ints_mo._matrices[1] is None

        with self.subTest("Alpha and Beta"):
            ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (mat_a, mat_b))
            ints_mo = ints_ao.transform_basis(transform)
            assert np.allclose(ints_mo._matrices[0], 4 * mat_a)
            assert np.allclose(ints_mo._matrices[1], 4 * mat_b)

        with self.subTest("Beta custom coeff with only alpha"):
            transform_beta = ElectronicBasisTransform(
                ElectronicBasis.AO, ElectronicBasis.MO, 2 * np.eye(2), 3 * np.eye(2)
            )
            ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (mat_a, None))
            ints_mo = ints_ao.transform_basis(transform_beta)
            assert np.allclose(ints_mo._matrices[0], 4 * mat_a)
            assert ints_mo._matrices[1] is None

        with self.subTest("Beta custom coeff"):
            transform_beta = ElectronicBasisTransform(
                ElectronicBasis.AO, ElectronicBasis.MO, 2 * np.eye(2), 3 * np.eye(2)
            )
            ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.AO, (mat_a, mat_b))
            ints_mo = ints_ao.transform_basis(transform_beta)
            assert np.allclose(ints_mo._matrices[0], 4 * mat_a)
            assert np.allclose(ints_mo._matrices[1], 9 * mat_b)

        with self.subTest("Final basis match"):
            ints_ao = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, None))
            ints_mo = ints_ao.transform_basis(transform)
            assert ints_ao == ints_mo

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
            assert np.allclose(
                mat_so, np.asarray([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4]])
            )

        with self.subTest("Alpha and beta"):
            ints = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, mat_b))
            mat_so = ints.to_spin()
            assert np.allclose(
                mat_so, np.asarray([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, -4, -3], [0, 0, -2, -1]])
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
                    ("NIII", 1),
                    ("+-II", 2),
                    ("-+II", -3),
                    ("INII", 4),
                    ("IINI", 1),
                    ("II+-", 2),
                    ("II-+", -3),
                    ("IIIN", 4),
                ],
            ):
                assert real_label == exp_label
                assert np.isclose(real_coeff, exp_coeff)

        with self.subTest("Alpha and beta"):
            ints = OneBodyElectronicIntegrals(ElectronicBasis.MO, (mat_a, mat_b))
            op = ints.to_second_q_op()
            for (real_label, real_coeff), (exp_label, exp_coeff) in zip(
                op.to_list(),
                [
                    ("NIII", 1),
                    ("+-II", 2),
                    ("-+II", -3),
                    ("INII", 4),
                    ("IINI", -4),
                    ("II+-", -3),
                    ("II-+", 2),
                    ("IIIN", -1),
                ],
            ):
                assert real_label == exp_label
                assert np.isclose(real_coeff, exp_coeff)
