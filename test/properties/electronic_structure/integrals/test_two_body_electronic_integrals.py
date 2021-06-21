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

"""Test TwoBodyElectronicIntegrals."""

import json
from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.properties.electronic_structure.bases import (
    ElectronicBasis,
    ElectronicBasisTransform,
)
from qiskit_nature.properties.electronic_structure.integrals import TwoBodyElectronicIntegrals


class TestTwoBodyElectronicIntegrals(QiskitNatureTestCase):
    """Test TwoBodyElectronicIntegrals."""

    def test_init(self):
        """Test construction."""
        random = np.random.rand(2, 2, 2, 2)

        with self.subTest("Normal"):
            TwoBodyElectronicIntegrals(ElectronicBasis.MO, (random, None, None, None))

        with self.subTest("Alpha and beta"):
            TwoBodyElectronicIntegrals(ElectronicBasis.MO, (random, random, random, random))

        with self.subTest("Alpha and beta but transpose last one"):
            TwoBodyElectronicIntegrals(ElectronicBasis.MO, (random, random, random, None))

        with self.subTest("Spin"):
            TwoBodyElectronicIntegrals(ElectronicBasis.SO, random)

        with self.subTest("Mismatching basis and number of matrices"):
            with self.assertRaises(TypeError):
                TwoBodyElectronicIntegrals(ElectronicBasis.MO, random)

        with self.subTest("Mismatching basis and number of matrices 2"):
            with self.assertRaises(TypeError):
                TwoBodyElectronicIntegrals(ElectronicBasis.SO, (random, None, None, None))

        with self.subTest("Missing alpha"):
            with self.assertRaises(TypeError):
                TwoBodyElectronicIntegrals(ElectronicBasis.MO, (None, random, random, random))

    def test_transform_basis(self):
        """Test transform_basis"""
        mat_aa = np.arange(16).reshape((2, 2, 2, 2))
        mat_ba = np.arange(16, 32).reshape((2, 2, 2, 2))
        mat_bb = np.arange(-16, 0).reshape((2, 2, 2, 2))

        transform = ElectronicBasisTransform(ElectronicBasis.AO, ElectronicBasis.MO, 2 * np.eye(2))

        with self.subTest("Pure Alpha"):
            ints_ao = TwoBodyElectronicIntegrals(ElectronicBasis.AO, (mat_aa, None, None, None))
            ints_mo = ints_ao.transform_basis(transform)
            assert np.allclose(ints_mo._matrices[0], 16 * mat_aa)
            assert ints_mo._matrices[1] is None
            assert ints_mo._matrices[2] is None
            assert ints_mo._matrices[3] is None

        with self.subTest("Alpha and Beta"):
            ints_ao = TwoBodyElectronicIntegrals(
                ElectronicBasis.AO, (mat_aa, mat_ba, mat_bb, mat_ba.T)
            )
            ints_mo = ints_ao.transform_basis(transform)
            assert np.allclose(ints_mo._matrices[0], 16 * mat_aa)
            assert np.allclose(ints_mo._matrices[1], 16 * mat_ba)
            assert np.allclose(ints_mo._matrices[2], 16 * mat_bb)
            assert np.allclose(ints_mo._matrices[3], 16 * mat_ba.T)

        with self.subTest("Beta custom coeff with only alpha"):
            transform_beta = ElectronicBasisTransform(
                ElectronicBasis.AO, ElectronicBasis.MO, 2 * np.eye(2), 3 * np.eye(2)
            )
            ints_ao = TwoBodyElectronicIntegrals(ElectronicBasis.AO, (mat_aa, None, None, None))
            ints_mo = ints_ao.transform_basis(transform_beta)
            assert np.allclose(ints_mo._matrices[0], 16 * mat_aa)
            assert ints_mo._matrices[1] is None
            assert ints_mo._matrices[2] is None
            assert ints_mo._matrices[3] is None

        with self.subTest("Beta custom coeff"):
            transform_beta = ElectronicBasisTransform(
                ElectronicBasis.AO, ElectronicBasis.MO, 2 * np.eye(2), 3 * np.eye(2)
            )
            ints_ao = TwoBodyElectronicIntegrals(
                ElectronicBasis.AO, (mat_aa, mat_ba, mat_bb, mat_ba.T)
            )
            ints_mo = ints_ao.transform_basis(transform_beta)
            assert np.allclose(ints_mo._matrices[0], 16 * mat_aa)
            assert np.allclose(ints_mo._matrices[1], 36 * mat_ba)
            assert np.allclose(ints_mo._matrices[2], 81 * mat_bb)
            assert np.allclose(ints_mo._matrices[3], 36 * mat_ba.T)

        with self.subTest("Final basis match"):
            ints_ao = TwoBodyElectronicIntegrals(ElectronicBasis.MO, (mat_aa, None, None, None))
            ints_mo = ints_ao.transform_basis(transform)
            assert ints_ao == ints_mo

        with self.subTest("Inital basis mismatch"):
            with self.assertRaises(QiskitNatureError):
                ints_ao = TwoBodyElectronicIntegrals(ElectronicBasis.SO, mat_aa)
                ints_ao.transform_basis(transform)

    def test_to_spin(self):
        """Test to_spin"""
        mat_aa = np.arange(16).reshape((2, 2, 2, 2))
        mat_ba = np.arange(16, 32).reshape((2, 2, 2, 2))
        mat_bb = np.arange(-16, 0).reshape((2, 2, 2, 2))

        with self.subTest("Only alpha"):
            ints = TwoBodyElectronicIntegrals(ElectronicBasis.MO, (mat_aa, None, None, None))
            mat_so = ints.to_spin()
            expected = np.fromfile(
                self.get_resource_path(
                    "two_body_test_to_spin_only_alpha_expected.numpy.bin",
                    "properties/electronic_structure/integrals/resources",
                )
            ).reshape((4, 4, 4, 4))
            assert np.allclose(mat_so, expected)

        with self.subTest("Alpha and beta"):
            ints = TwoBodyElectronicIntegrals(
                ElectronicBasis.MO, (mat_aa, mat_ba, mat_bb, mat_ba.T)
            )
            mat_so = ints.to_spin()
            expected = np.fromfile(
                self.get_resource_path(
                    "two_body_test_to_spin_alpha_and_beta_expected.numpy.bin",
                    "properties/electronic_structure/integrals/resources",
                )
            ).reshape((4, 4, 4, 4))
            assert np.allclose(mat_so, expected)

    def test_to_second_q_op(self):
        """Test to_second_q_op"""
        mat_aa = np.arange(16).reshape((2, 2, 2, 2))
        mat_ba = np.arange(16, 32).reshape((2, 2, 2, 2))
        mat_bb = np.arange(-16, 0).reshape((2, 2, 2, 2))

        with self.subTest("Only alpha"):
            ints = TwoBodyElectronicIntegrals(ElectronicBasis.MO, (mat_aa, None, None, None))
            op = ints.to_second_q_op()
            with open(
                self.get_resource_path(
                    "two_body_test_to_second_q_op_only_alpha_expected.json",
                    "properties/electronic_structure/integrals/resources",
                ),
                "r",
            ) as file:
                expected = json.load(file)
            for (real_label, real_coeff), (exp_label, exp_coeff) in zip(op.to_list(), expected):
                assert real_label == exp_label
                assert np.isclose(real_coeff, exp_coeff)

        with self.subTest("Alpha and beta"):
            ints = TwoBodyElectronicIntegrals(
                ElectronicBasis.MO, (mat_aa, mat_ba, mat_bb, mat_ba.T)
            )
            op = ints.to_second_q_op()
            with open(
                self.get_resource_path(
                    "two_body_test_to_second_q_op_alpha_and_beta_expected.json",
                    "properties/electronic_structure/integrals/resources",
                ),
                "r",
            ) as file:
                expected = json.load(file)
            for (real_label, real_coeff), (exp_label, exp_coeff) in zip(op.to_list(), expected):
                assert real_label == exp_label
                assert np.isclose(real_coeff, exp_coeff)
