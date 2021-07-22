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

"""Tests for the ActiveSpaceTransformer."""

import unittest

from test import QiskitNatureTestCase

from ddt import ddt, idata, unpack
import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers.second_quantization import HDF5Driver, QMolecule
from qiskit_nature.transformers import ActiveSpaceTransformer


@ddt
class TestActiveSpaceTransformer(QiskitNatureTestCase):
    """ActiveSpaceTransformer tests."""

    def assertQMolecule(self, q_molecule, expected, dict_key="ActiveSpaceTransformer"):
        """Asserts that the two `QMolecule object's relevant fields are equivalent."""
        with self.subTest("MO 1-electron integrals"):
            np.testing.assert_array_almost_equal(q_molecule.mo_onee_ints, expected.mo_onee_ints)
        with self.subTest("MO 2-electron integrals"):
            np.testing.assert_array_almost_equal(q_molecule.mo_eri_ints, expected.mo_eri_ints)
        with self.subTest("Inactive energy"):
            self.assertAlmostEqual(
                q_molecule.energy_shift[dict_key],
                expected.energy_shift["ActiveSpaceTransformer"],
            )

        with self.subTest("MO 1-electron x dipole integrals"):
            np.testing.assert_array_almost_equal(q_molecule.x_dip_mo_ints, expected.x_dip_mo_ints)
        with self.subTest("X dipole energy shift"):
            self.assertAlmostEqual(
                q_molecule.x_dip_energy_shift[dict_key],
                expected.x_dip_energy_shift["ActiveSpaceTransformer"],
            )
        with self.subTest("MO 1-electron y dipole integrals"):
            np.testing.assert_array_almost_equal(q_molecule.y_dip_mo_ints, expected.y_dip_mo_ints)
        with self.subTest("Y dipole energy shift"):
            self.assertAlmostEqual(
                q_molecule.y_dip_energy_shift[dict_key],
                expected.y_dip_energy_shift["ActiveSpaceTransformer"],
            )
        with self.subTest("MO 1-electron z dipole integrals"):
            np.testing.assert_array_almost_equal(q_molecule.z_dip_mo_ints, expected.z_dip_mo_ints)
        with self.subTest("Z dipole energy shift"):
            self.assertAlmostEqual(
                q_molecule.z_dip_energy_shift[dict_key],
                expected.z_dip_energy_shift["ActiveSpaceTransformer"],
            )

    @idata(
        [
            {"num_electrons": 2, "num_molecular_orbitals": 2},
        ]
    )
    def test_full_active_space(self, kwargs):
        """Test that transformer has no effect when all orbitals are active."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("H2_sto3g.hdf5", "transformers/second_quantization")
        )
        q_molecule = driver.run()

        q_molecule.energy_shift["ActiveSpaceTransformer"] = 0.0
        q_molecule.x_dip_energy_shift["ActiveSpaceTransformer"] = 0.0
        q_molecule.y_dip_energy_shift["ActiveSpaceTransformer"] = 0.0
        q_molecule.z_dip_energy_shift["ActiveSpaceTransformer"] = 0.0

        trafo = ActiveSpaceTransformer(**kwargs)
        q_molecule_reduced = trafo.transform(q_molecule)

        self.assertQMolecule(q_molecule_reduced, q_molecule)

    def test_minimal_active_space(self):
        """Test a minimal active space manually."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("H2_631g.hdf5", "transformers/second_quantization")
        )
        q_molecule = driver.run()

        trafo = ActiveSpaceTransformer(num_electrons=2, num_molecular_orbitals=2)
        q_molecule_reduced = trafo.transform(q_molecule)

        expected = QMolecule()
        expected.mo_onee_ints = np.asarray([[-1.24943841, 0.0], [0.0, -0.547816138]])
        expected.mo_eri_ints = np.asarray(
            [
                [
                    [[0.652098466, 0.0], [0.0, 0.433536565]],
                    [[0.0, 0.0794483182], [0.0794483182, 0.0]],
                ],
                [
                    [[0.0, 0.0794483182], [0.0794483182, 0.0]],
                    [[0.433536565, 0.0], [0.0, 0.385524695]],
                ],
            ]
        )

        expected.x_dip_mo_ints = np.zeros((2, 2))
        expected.y_dip_mo_ints = np.zeros((2, 2))
        expected.z_dip_mo_ints = np.asarray([[0.69447435, -1.01418298], [-1.01418298, 0.69447435]])

        expected.energy_shift["ActiveSpaceTransformer"] = 0.0
        expected.x_dip_energy_shift["ActiveSpaceTransformer"] = 0.0
        expected.y_dip_energy_shift["ActiveSpaceTransformer"] = 0.0
        expected.z_dip_energy_shift["ActiveSpaceTransformer"] = 0.0

        self.assertQMolecule(q_molecule_reduced, expected)

    def test_unpaired_electron_active_space(self):
        """Test an active space with an unpaired electron."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("BeH_sto3g.hdf5", "transformers/second_quantization")
        )
        q_molecule = driver.run()

        trafo = ActiveSpaceTransformer(num_electrons=(2, 1), num_molecular_orbitals=3)
        q_molecule_reduced = trafo.transform(q_molecule)

        expected = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "BeH_sto3g_reduced.hdf5", "transformers/second_quantization"
            )
        ).run()

        self.assertQMolecule(q_molecule_reduced, expected)

    def test_arbitrary_active_orbitals(self):
        """Test manual selection of active orbital indices."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("H2_631g.hdf5", "transformers/second_quantization")
        )
        q_molecule = driver.run()

        trafo = ActiveSpaceTransformer(
            num_electrons=2, num_molecular_orbitals=2, active_orbitals=[0, 2]
        )
        q_molecule_reduced = trafo.transform(q_molecule)

        expected = QMolecule()
        expected.mo_onee_ints = np.asarray([[-1.24943841, -0.16790838], [-0.16790838, -0.18307469]])
        expected.mo_eri_ints = np.asarray(
            [
                [
                    [[0.65209847, 0.16790822], [0.16790822, 0.53250905]],
                    [[0.16790822, 0.10962908], [0.10962908, 0.11981429]],
                ],
                [
                    [[0.16790822, 0.10962908], [0.10962908, 0.11981429]],
                    [[0.53250905, 0.11981429], [0.11981429, 0.46345617]],
                ],
            ]
        )

        expected.x_dip_mo_ints = np.zeros((2, 2))
        expected.y_dip_mo_ints = np.zeros((2, 2))
        expected.z_dip_mo_ints = np.asarray([[0.69447435, 0.0], [0.0, 0.69447435]])

        expected.energy_shift["ActiveSpaceTransformer"] = 0.0
        expected.x_dip_energy_shift["ActiveSpaceTransformer"] = 0.0
        expected.y_dip_energy_shift["ActiveSpaceTransformer"] = 0.0
        expected.z_dip_energy_shift["ActiveSpaceTransformer"] = 0.0

        self.assertQMolecule(q_molecule_reduced, expected)

    @idata(
        [
            [2, 3, None, "More active orbitals requested than available in total."],
            [4, 2, None, "More active electrons requested than available in total."],
            [(1, 0), 2, None, "The number of inactive electrons may not be odd."],
            [2, 2, [0, 1, 2], "The number of active orbitals do not match."],
            [2, 2, [1, 2], "The number of active electrons do not match."],
            [1, 2, None, "The number of active electrons must be even when not a tuple."],
            [-2, 2, None, "The number of active electrons must not be negative."],
        ]
    )
    @unpack
    def test_error_raising(self, num_electrons, num_molecular_orbitals, active_orbitals, message):
        """Test errors are being raised in certain scenarios."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path("H2_sto3g.hdf5", "transformers/second_quantization")
        )
        q_molecule = driver.run()

        with self.assertRaises(QiskitNatureError, msg=message):
            ActiveSpaceTransformer(
                num_electrons=num_electrons,
                num_molecular_orbitals=num_molecular_orbitals,
                active_orbitals=active_orbitals,
            ).transform(q_molecule)

    def test_active_space_for_q_molecule_v2(self):
        """Test based on QMolecule v2 (mo_occ not available)."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "H2_sto3g_v2.hdf5", "transformers/second_quantization"
            )
        )
        q_molecule = driver.run()

        q_molecule.energy_shift["ActiveSpaceTransformer"] = 0.0
        q_molecule.x_dip_energy_shift["ActiveSpaceTransformer"] = 0.0
        q_molecule.y_dip_energy_shift["ActiveSpaceTransformer"] = 0.0
        q_molecule.z_dip_energy_shift["ActiveSpaceTransformer"] = 0.0

        trafo = ActiveSpaceTransformer(num_electrons=2, num_molecular_orbitals=2)
        q_molecule_reduced = trafo.transform(q_molecule)

        self.assertQMolecule(q_molecule_reduced, q_molecule)


if __name__ == "__main__":
    unittest.main()
