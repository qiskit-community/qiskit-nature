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

"""Test VibrationalEnergy Property"""

import json
import tempfile
import warnings
from test import QiskitNatureTestCase

import h5py
import numpy as np

from qiskit_nature.drivers import WatsonHamiltonian
from qiskit_nature.properties.second_quantization.vibrational import VibrationalEnergy
from qiskit_nature.properties.second_quantization.vibrational.bases import HarmonicBasis
from qiskit_nature.properties.second_quantization.vibrational.integrals import VibrationalIntegrals


class TestVibrationalEnergy(QiskitNatureTestCase):
    """Test VibrationalEnergy Property"""

    def setUp(self):
        """Setup basis."""
        super().setUp()
        basis = HarmonicBasis([2, 2, 2, 2])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            watson = WatsonHamiltonian(
                [
                    [352.3005875, 2, 2],
                    [-352.3005875, -2, -2],
                    [631.6153975, 1, 1],
                    [-631.6153975, -1, -1],
                    [115.653915, 4, 4],
                    [-115.653915, -4, -4],
                    [115.653915, 3, 3],
                    [-115.653915, -3, -3],
                    [-15.341901966295344, 2, 2, 2],
                    [-88.2017421687633, 1, 1, 2],
                    [42.40478531359112, 4, 4, 2],
                    [26.25167512727164, 4, 3, 2],
                    [2.2874639206341865, 3, 3, 2],
                    [0.4207357291666667, 2, 2, 2, 2],
                    [4.9425425, 1, 1, 2, 2],
                    [1.6122932291666665, 1, 1, 1, 1],
                    [-4.194299375, 4, 4, 2, 2],
                    [-4.194299375, 3, 3, 2, 2],
                    [-10.20589125, 4, 4, 1, 1],
                    [-10.20589125, 3, 3, 1, 1],
                    [2.2973803125, 4, 4, 4, 4],
                    [2.7821204166666664, 4, 4, 4, 3],
                    [7.329224375, 4, 4, 3, 3],
                    [-2.7821200000000004, 4, 3, 3, 3],
                    [2.2973803125, 3, 3, 3, 3],
                ],
                4,
            )
        self.prop = VibrationalEnergy.from_legacy_driver_result(watson)
        self.prop.basis = basis

    def test_second_q_ops(self):
        """Test second_q_ops."""
        ops = self.prop.second_q_ops()
        self.assertEqual(len(ops), 1)
        with open(
            self.get_resource_path(
                "vibrational_energy_op.json", "properties/second_quantization/vibrational/resources"
            ),
            "r",
            encoding="utf8",
        ) as file:
            expected = json.load(file)
        for op, expected_op in zip(ops[0].to_list(), expected):
            self.assertEqual(op[0], expected_op[0])
            self.assertTrue(np.isclose(op[1], expected_op[1]))

    def test_to_hdf5(self):
        """Test to_hdf5."""
        with tempfile.TemporaryFile() as tmp_file:
            with h5py.File(tmp_file, "w") as file:
                self.prop.to_hdf5(file)

            with h5py.File(tmp_file, "r") as file:
                count = 0

                for name, group in file.items():
                    count += 1
                    self.assertEqual(name, "VibrationalEnergy")

                    expected_1 = [
                        [352.3005875, 2, 2],
                        [-352.3005875, -2, -2],
                        [631.6153975, 1, 1],
                        [-631.6153975, -1, -1],
                        [115.653915, 4, 4],
                        [-115.653915, -4, -4],
                        [115.653915, 3, 3],
                        [-115.653915, -3, -3],
                        [-15.341901966295344, 2, 2, 2],
                        [0.4207357291666667, 2, 2, 2, 2],
                        [1.6122932291666665, 1, 1, 1, 1],
                        [2.2973803125, 4, 4, 4, 4],
                        [2.2973803125, 3, 3, 3, 3],
                        [-88.2017421687633, 1, 1, 2],
                        [42.40478531359112, 4, 4, 2],
                        [26.25167512727164, 4, 3, 2],
                        [2.2874639206341865, 3, 3, 2],
                        [4.9425425, 1, 1, 2, 2],
                        [-4.194299375, 4, 4, 2, 2],
                        [-4.194299375, 3, 3, 2, 2],
                        [-10.20589125, 4, 4, 1, 1],
                        [-10.20589125, 3, 3, 1, 1],
                        [2.7821204166666664, 4, 4, 4, 3],
                        [7.329224375, 4, 4, 3, 3],
                        [-2.7821200000000004, 4, 3, 3, 3],
                    ]
                    one_body = VibrationalIntegrals.from_hdf5(
                        group["vibrational_integrals"]["1BodyVibrationalIntegrals"]
                    )
                    for exp_int, real_int in zip(expected_1, one_body._integrals):
                        count += 1
                        self.assertAlmostEqual(exp_int[0], real_int[0])
                        self.assertEqual(tuple(exp_int[1:]), real_int[1])

                    expected_2 = [
                        [-88.2017421687633, 1, 1, 2],
                        [42.40478531359112, 4, 4, 2],
                        [2.2874639206341865, 3, 3, 2],
                        [4.9425425, 1, 1, 2, 2],
                        [-4.194299375, 4, 4, 2, 2],
                        [-4.194299375, 3, 3, 2, 2],
                        [-10.20589125, 4, 4, 1, 1],
                        [-10.20589125, 3, 3, 1, 1],
                        [2.7821204166666664, 4, 4, 4, 3],
                        [7.329224375, 4, 4, 3, 3],
                        [-2.7821200000000004, 4, 3, 3, 3],
                    ]
                    two_body = VibrationalIntegrals.from_hdf5(
                        group["vibrational_integrals"]["2BodyVibrationalIntegrals"]
                    )
                    for exp_int, real_int in zip(expected_2, two_body._integrals):
                        count += 1
                        self.assertAlmostEqual(exp_int[0], real_int[0])
                        self.assertEqual(tuple(exp_int[1:]), real_int[1])

                    expected_3 = [
                        [26.25167512727164, 4, 3, 2],
                    ]
                    three_body = VibrationalIntegrals.from_hdf5(
                        group["vibrational_integrals"]["3BodyVibrationalIntegrals"]
                    )
                    for exp_int, real_int in zip(expected_3, three_body._integrals):
                        count += 1
                        self.assertAlmostEqual(exp_int[0], real_int[0])
                        self.assertEqual(tuple(exp_int[1:]), real_int[1])

                self.assertEqual(count, 26)

    def test_from_hdf5(self):
        """Test from_hdf5."""
        self.skipTest("Testing via VibrationalStructureResult tests.")
