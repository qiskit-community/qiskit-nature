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

"""Test HarmonicBasis."""

import json
import tempfile
from test import QiskitNatureTestCase
from ddt import ddt, data, unpack

import h5py
import numpy as np

from qiskit_nature.second_q.properties.bases import HarmonicBasis
from qiskit_nature.second_q.properties.integrals import (
    VibrationalIntegrals,
)


@ddt
class TestHarmonicBasis(QiskitNatureTestCase):
    """Test HarmonicBasis."""

    @unpack
    @data(
        (
            1,
            [
                (352.3005875, (2, 2)),
                (-352.3005875, (-2, -2)),
                (631.6153975, (1, 1)),
                (-631.6153975, (-1, -1)),
                (115.653915, (4, 4)),
                (-115.653915, (-4, -4)),
                (115.653915, (3, 3)),
                (-115.653915, (-3, -3)),
                (-15.341901966295344, (2, 2, 2)),
                (0.4207357291666667, (2, 2, 2, 2)),
                (1.6122932291666665, (1, 1, 1, 1)),
                (2.2973803125, (4, 4, 4, 4)),
                (2.2973803125, (3, 3, 3, 3)),
            ],
        ),
        (
            2,
            [
                (-88.2017421687633, (1, 1, 2)),
                (42.40478531359112, (4, 4, 2)),
                (2.2874639206341865, (3, 3, 2)),
                (4.9425425, (1, 1, 2, 2)),
                (-4.194299375, (4, 4, 2, 2)),
                (-4.194299375, (3, 3, 2, 2)),
                (-10.20589125, (4, 4, 1, 1)),
                (-10.20589125, (3, 3, 1, 1)),
                (2.7821204166666664, (4, 4, 4, 3)),
                (7.329224375, (4, 4, 3, 3)),
                (-2.7821200000000004, (4, 3, 3, 3)),
            ],
        ),
        (
            3,
            [
                (26.25167512727164, (4, 3, 2)),
            ],
        ),
    )
    def test_harmonic_basis(self, num_body, integrals):
        """Test HarmonicBasis"""
        integrals = VibrationalIntegrals(num_body, integrals)

        # TODO: test more variants
        num_modes = 4
        num_modals = 2
        num_modals_per_mode = [num_modals] * num_modes
        basis = HarmonicBasis(num_modals_per_mode)

        integrals.basis = basis

        with self.subTest("Test to_basis"):
            matrix = integrals.to_basis()
            nonzero = np.nonzero(matrix)

            with open(
                self.get_resource_path(
                    f"test_harmonic_basis_to_basis_{num_body}.json",
                    "second_q/properties/bases/resources",
                ),
                "r",
                encoding="utf8",
            ) as file:
                exp_nonzero, exp_values = json.load(file)

            self.assertTrue(np.allclose(np.asarray(nonzero), np.asarray(exp_nonzero)))
            self.assertTrue(np.allclose(matrix[nonzero], np.asarray(exp_values)))

        with self.subTest("Test to_second_q_op"):
            op = integrals.to_second_q_op()

            with open(
                self.get_resource_path(
                    f"test_harmonic_basis_to_second_q_op_{num_body}.json",
                    "second_q/properties/bases/resources",
                ),
                "r",
                encoding="utf8",
            ) as file:
                operator = json.load(file)

            for (real_label, real_coeff), (exp_label, exp_coeff) in zip(op.to_list(), operator):
                self.assertEqual(real_label, exp_label)
                self.assertTrue(np.isclose(real_coeff, exp_coeff))

    def test_to_hdf5(self):
        """Test to_hdf5."""
        num_modes = 4
        num_modals = 2
        num_modals_per_mode = [num_modals] * num_modes
        basis = HarmonicBasis(num_modals_per_mode)

        with tempfile.TemporaryFile() as tmp_file:
            with h5py.File(tmp_file, "w") as file:
                basis.to_hdf5(file)

    def test_from_hdf5(self):
        """Test from_hdf5."""
        num_modes = 4
        num_modals = 2
        num_modals_per_mode = [num_modals] * num_modes
        basis = HarmonicBasis(num_modals_per_mode)

        with tempfile.TemporaryFile() as tmp_file:
            with h5py.File(tmp_file, "w") as file:
                basis.to_hdf5(file)

            with h5py.File(tmp_file, "r") as file:
                read_prop = HarmonicBasis.from_hdf5(file["HarmonicBasis"])

                self.assertTrue(
                    np.allclose(basis.num_modals_per_mode, read_prop.num_modals_per_mode)
                )
