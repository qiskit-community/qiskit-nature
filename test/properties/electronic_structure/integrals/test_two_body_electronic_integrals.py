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
            with self.assertRaises(AssertionError):
                TwoBodyElectronicIntegrals(ElectronicBasis.MO, random)

        with self.subTest("Mismatching basis and number of matrices 2"):
            with self.assertRaises(AssertionError):
                TwoBodyElectronicIntegrals(ElectronicBasis.SO, (random, None, None, None))

        with self.subTest("Missing alpha"):
            with self.assertRaises(AssertionError):
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
            expected = (
                [
                    [
                        [
                            [0.0, -1.0, 0.0, 0.0],
                            [-4.0, -5.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [-2.0, -3.0, 0.0, 0.0],
                            [-6.0, -7.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [-4.0, -5.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [-2.0, -3.0, 0.0, 0.0],
                            [-6.0, -7.0, 0.0, 0.0],
                        ],
                    ],
                    [
                        [
                            [-0.5, -1.5, 0.0, 0.0],
                            [-4.5, -5.5, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [-2.5, -3.5, 0.0, 0.0],
                            [-6.5, -7.5, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [-0.5, -1.5, 0.0, 0.0],
                            [-4.5, -5.5, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [-2.5, -3.5, 0.0, 0.0],
                            [-6.5, -7.5, 0.0, 0.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 0.0, 0.0, -1.0],
                            [0.0, 0.0, -4.0, -5.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, -2.0, -3.0],
                            [0.0, 0.0, -6.0, -7.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, -1.0],
                            [0.0, 0.0, -4.0, -5.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, -2.0, -3.0],
                            [0.0, 0.0, -6.0, -7.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 0.0, -0.5, -1.5],
                            [0.0, 0.0, -4.5, -5.5],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, -2.5, -3.5],
                            [0.0, 0.0, -6.5, -7.5],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, -0.5, -1.5],
                            [0.0, 0.0, -4.5, -5.5],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, -2.5, -3.5],
                            [0.0, 0.0, -6.5, -7.5],
                        ],
                    ],
                ],
            )
            assert np.allclose(mat_so, expected)

        with self.subTest("Alpha and beta"):
            ints = TwoBodyElectronicIntegrals(
                ElectronicBasis.MO, (mat_aa, mat_ba, mat_bb, mat_ba.T)
            )
            mat_so = ints.to_spin()
            expected = (
                [
                    [
                        [
                            [0.0, -1.0, 0.0, 0.0],
                            [-4.0, -5.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [-2.0, -3.0, 0.0, 0.0],
                            [-6.0, -7.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [-8.0, -9.0, 0.0, 0.0],
                            [-12.0, -13.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [-10.0, -11.0, 0.0, 0.0],
                            [-14.0, -15.0, 0.0, 0.0],
                        ],
                    ],
                    [
                        [
                            [-0.5, -1.5, 0.0, 0.0],
                            [-4.5, -5.5, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [-2.5, -3.5, 0.0, 0.0],
                            [-6.5, -7.5, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [-8.5, -9.5, 0.0, 0.0],
                            [-12.5, -13.5, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [-10.5, -11.5, 0.0, 0.0],
                            [-14.5, -15.5, 0.0, 0.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 0.0, -8.0, -10.0],
                            [0.0, 0.0, -8.5, -10.5],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, -9.0, -11.0],
                            [0.0, 0.0, -9.5, -11.5],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 8.0, 7.0],
                            [0.0, 0.0, 4.0, 3.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 6.0, 5.0],
                            [0.0, 0.0, 2.0, 1.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 0.0, -12.0, -14.0],
                            [0.0, 0.0, -12.5, -14.5],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, -13.0, -15.0],
                            [0.0, 0.0, -13.5, -15.5],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 7.5, 6.5],
                            [0.0, 0.0, 3.5, 2.5],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 5.5, 4.5],
                            [0.0, 0.0, 1.5, 0.5],
                        ],
                    ],
                ],
            )
            assert np.allclose(mat_so, expected)

    def test_to_second_q_op(self):
        """Test to_second_q_op"""
        mat_aa = np.arange(16).reshape((2, 2, 2, 2))
        mat_ba = np.arange(16, 32).reshape((2, 2, 2, 2))
        mat_bb = np.arange(-16, 0).reshape((2, 2, 2, 2))

        with self.subTest("Only alpha"):
            ints = TwoBodyElectronicIntegrals(ElectronicBasis.MO, (mat_aa, None, None, None))
            op = ints.to_second_q_op()
            for (real_label, real_coeff), (exp_label, exp_coeff) in zip(
                op.to_list(),
                [
                    ("NNII", -5),
                    ("NNII", 6),
                    ("+-NI", 1),
                    ("NI-+", -4),
                    ("+--+", -5),
                    ("NI+-", 2),
                    ("+-+-", 3),
                    ("NIIN", 6),
                    ("+-IN", 7),
                    ("NNII", 1.5),
                    ("NNII", -2.5),
                    ("-+NI", -0.5),
                    ("INNI", 1.5),
                    ("-+-+", 4.5),
                    ("IN-+", -5.5),
                    ("-++-", -2.5),
                    ("IN+-", 3.5),
                    ("-+IN", -6.5),
                    ("ININ", 7.5),
                    ("NI+-", 1),
                    ("-+NI", -4),
                    ("-++-", -5),
                    ("+-NI", 2),
                    ("+-+-", 3),
                    ("INNI", 6),
                    ("IN+-", 7),
                    ("IINN", -5),
                    ("IINN", 6),
                    ("NI-+", -0.5),
                    ("NIIN", 1.5),
                    ("-+-+", 4.5),
                    ("-+IN", -5.5),
                    ("+--+", -2.5),
                    ("+-IN", 3.5),
                    ("IN-+", -6.5),
                    ("ININ", 7.5),
                    ("IINN", 1.5),
                    ("IINN", -2.5),
                ],
            ):
                assert real_label == exp_label
                assert np.isclose(real_coeff, exp_coeff)

        with self.subTest("Alpha and beta"):
            ints = TwoBodyElectronicIntegrals(
                ElectronicBasis.MO, (mat_aa, mat_ba, mat_bb, mat_ba.T)
            )
            op = ints.to_second_q_op()
            for (real_label, real_coeff), (exp_label, exp_coeff) in zip(
                op.to_list(),
                [
                    ("NNII", -5),
                    ("NNII", 6),
                    ("NINI", 8),
                    ("+-NI", 9),
                    ("NI-+", -12),
                    ("+--+", -13),
                    ("NI+-", 10),
                    ("+-+-", 11),
                    ("NIIN", 14),
                    ("+-IN", 15),
                    ("NNII", 1.5),
                    ("NNII", -2.5),
                    ("-+NI", -8.5),
                    ("INNI", 9.5),
                    ("-+-+", 12.5),
                    ("IN-+", -13.5),
                    ("-++-", -10.5),
                    ("IN+-", 11.5),
                    ("-+IN", -14.5),
                    ("ININ", 15.5),
                    ("NINI", 8),
                    ("NI+-", 10),
                    ("-+NI", -8.5),
                    ("-++-", -10.5),
                    ("+-NI", 9),
                    ("+-+-", 11),
                    ("INNI", 9.5),
                    ("IN+-", 11.5),
                    ("IINN", 3),
                    ("IINN", -2),
                    ("NI-+", -12),
                    ("NIIN", 14),
                    ("-+-+", 12.5),
                    ("-+IN", -14.5),
                    ("+--+", -13),
                    ("+-IN", 15),
                    ("IN-+", -13.5),
                    ("ININ", 15.5),
                    ("IINN", -6.5),
                    ("IINN", 5.5),
                ],
            ):
                assert real_label == exp_label
                assert np.isclose(real_coeff, exp_coeff)
