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

"""Tests for the ActiveSpaceTransformer."""

import unittest

from test import QiskitNatureTestCase

from ddt import ddt, idata, unpack
import numpy as np

import qiskit_nature.optionals as _optionals
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.qcschema_translator import qcschema_to_problem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.properties import ElectronicDensity, ElectronicDipoleMoment
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer


@ddt
class TestActiveSpaceTransformer(QiskitNatureTestCase):
    """ActiveSpaceTransformer tests."""

    def assertDriverResult(self, driver_result, expected):
        """Asserts that the two `DriverResult` object's relevant fields are equivalent."""
        electronic_energy = driver_result.hamiltonian
        electronic_energy_exp = expected.hamiltonian
        with self.subTest("MO 1-electron integrals"):
            np.testing.assert_array_almost_equal(
                np.abs(electronic_energy.electronic_integrals.second_q_coeffs()["+-"]),
                np.abs(electronic_energy_exp.electronic_integrals.second_q_coeffs()["+-"]),
            )
        with self.subTest("MO 2-electron integrals"):
            np.testing.assert_array_almost_equal(
                np.abs(electronic_energy.electronic_integrals.second_q_coeffs()["++--"]),
                np.abs(electronic_energy_exp.electronic_integrals.second_q_coeffs()["++--"]),
            )
        with self.subTest("Inactive energy"):
            for key in electronic_energy_exp.constants.keys():
                self.assertAlmostEqual(
                    electronic_energy.constants[key],
                    electronic_energy_exp.constants[key],
                )

        if expected.properties.electronic_dipole_moment is not None:
            dip_moment = driver_result.properties.electronic_dipole_moment
            exp_moment = expected.properties.electronic_dipole_moment
            with self.subTest("Integrals"):
                for dipole, dipole_exp in zip(
                    (dip_moment.x_dipole, dip_moment.y_dipole, dip_moment.z_dipole),
                    (exp_moment.x_dipole, exp_moment.y_dipole, exp_moment.z_dipole),
                ):
                    np.testing.assert_array_almost_equal(
                        np.abs(dipole.second_q_coeffs()["+-"]),
                        np.abs(dipole_exp.second_q_coeffs()["+-"]),
                    )
            with self.subTest("Dipole shift"):
                for key in exp_moment.constants.keys():
                    np.testing.assert_array_almost_equal(
                        dip_moment.constants[key],
                        exp_moment.constants[key],
                    )

        if expected.properties.electronic_density is not None:
            density = driver_result.properties.electronic_density
            expected_density = expected.properties.electronic_density
            with self.subTest("ElectronicDensity"):
                self.assertTrue(density.equiv(expected_density))

        with self.subTest("attributes"):
            self.assertEqual(driver_result.num_particles, expected.num_particles)
            self.assertEqual(driver_result.num_spatial_orbitals, expected.num_spatial_orbitals)
            if expected.orbital_energies is not None:
                self.assertTrue(
                    np.allclose(driver_result.orbital_energies, expected.orbital_energies)
                )
            if expected.orbital_energies_b is not None:
                self.assertTrue(
                    np.allclose(driver_result.orbital_energies_b, expected.orbital_energies_b)
                )
            if expected.orbital_occupations is not None:
                self.assertTrue(
                    np.allclose(driver_result.orbital_occupations, expected.orbital_occupations)
                )
            if expected.orbital_occupations_b is not None:
                self.assertTrue(
                    np.allclose(driver_result.orbital_occupations_b, expected.orbital_occupations_b)
                )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    @idata(
        [
            {"num_electrons": 2, "num_spatial_orbitals": 2},
        ]
    )
    def test_full_active_space(self, kwargs):
        """test that transformer has no effect when all orbitals are active."""
        driver = PySCFDriver()
        driver_result = driver.run()

        driver_result.hamiltonian.constants["ActiveSpaceTransformer"] = 0.0
        driver_result.properties.electronic_dipole_moment.constants["ActiveSpaceTransformer"] = (
            0.0,
            0.0,
            0.0,
        )

        trafo = ActiveSpaceTransformer(**kwargs)
        driver_result_reduced = trafo.transform(driver_result)

        self.assertDriverResult(driver_result_reduced, driver_result)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_minimal_active_space(self):
        """Test a minimal active space manually."""
        driver = PySCFDriver(basis="631g")
        driver_result = driver.run()
        driver_result.properties.electronic_density = ElectronicDensity.from_orbital_occupation(
            driver_result.orbital_occupations,
            driver_result.orbital_occupations_b,
        )

        trafo = ActiveSpaceTransformer(2, 2)
        driver_result_reduced = trafo.transform(driver_result)

        electronic_energy = ElectronicEnergy.from_raw_integrals(
            np.asarray([[-1.24943841, 0.0], [0.0, -0.547816138]]),
            np.asarray(
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
            ),
        )
        electronic_energy.constants["ActiveSpaceTransformer"] = 0.0
        expected = ElectronicStructureProblem(electronic_energy)
        expected.num_particles = (1, 1)
        expected.num_spatial_orbitals = 2
        dipole_moment = ElectronicDipoleMoment(
            ElectronicIntegrals.from_raw_integrals(np.zeros((2, 2))),
            ElectronicIntegrals.from_raw_integrals(np.zeros((2, 2))),
            ElectronicIntegrals.from_raw_integrals(
                np.asarray([[0.69447435, -1.01418298], [-1.01418298, 0.69447435]]),
            ),
        )
        dipole_moment.constants["ActiveSpaceTransformer"] = (0.0, 0.0, 0.0)
        expected.properties.electronic_dipole_moment = dipole_moment
        density = ElectronicDensity.from_orbital_occupation([1, 0], [1, 0])
        expected.properties.electronic_density = density

        self.assertDriverResult(driver_result_reduced, expected)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_unpaired_electron_active_space(self):
        """Test an active space with an unpaired electron."""
        driver = PySCFDriver(atom="Be 0 0 0; H 0 0 1.3", basis="sto3g", spin=1)
        driver_result = driver.run()

        nelec = (2, 1)
        norb = 3
        trafo = ActiveSpaceTransformer(nelec, norb)
        driver_result_reduced = trafo.transform(driver_result)

        expected = qcschema_to_problem(
            QCSchema.from_json(
                self.get_resource_path("BeH_sto3g_reduced.json", "second_q/transformers/resources")
            ),
            include_dipole=False,
        )
        # add energy shift, which currently cannot be stored in the QCSchema
        expected.hamiltonian.constants["ActiveSpaceTransformer"] = -14.253802923103054
        expected.num_particles = nelec
        expected.num_spatial_orbitals = norb

        self.assertDriverResult(driver_result_reduced, expected)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_arbitrary_active_orbitals(self):
        """Test manual selection of active orbital indices."""
        driver = PySCFDriver(basis="631g")
        driver_result = driver.run()

        trafo = ActiveSpaceTransformer(2, 2, [0, 2])
        driver_result_reduced = trafo.transform(driver_result)

        electronic_energy = ElectronicEnergy.from_raw_integrals(
            np.asarray([[-1.24943841, -0.16790838], [-0.16790838, -0.18307469]]),
            np.asarray(
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
            ),
        )
        electronic_energy.constants["ActiveSpaceTransformer"] = 0.0
        expected = ElectronicStructureProblem(electronic_energy)
        expected.num_particles = (1, 1)
        expected.num_spatial_orbitals = 2
        dipole_moment = ElectronicDipoleMoment(
            ElectronicIntegrals.from_raw_integrals(np.zeros((2, 2))),
            ElectronicIntegrals.from_raw_integrals(np.zeros((2, 2))),
            ElectronicIntegrals.from_raw_integrals(
                np.asarray([[0.69447435, 0.0], [0.0, 0.69447435]]),
            ),
        )
        dipole_moment.constants["ActiveSpaceTransformer"] = (0.0, 0.0, 0.0)
        expected.properties.electronic_dipole_moment = dipole_moment
        self.assertDriverResult(driver_result_reduced, expected)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
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
    def test_error_raising(self, num_electrons, num_spatial_orbitals, active_orbitals, message):
        """Test errors are being raised in certain scenarios."""
        driver = PySCFDriver()
        driver_result = driver.run()

        with self.assertRaises(QiskitNatureError, msg=message):
            ActiveSpaceTransformer(
                num_electrons,
                num_spatial_orbitals,
                active_orbitals,
            ).transform(driver_result)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_tuple_num_electrons_with_manual_orbitals(self):
        """Regression test against https://github.com/Qiskit/qiskit-nature/issues/434."""
        driver = PySCFDriver(basis="631g")
        driver_result = driver.run()

        nelec = (1, 1)
        norb = 2
        trafo = ActiveSpaceTransformer(nelec, norb, [0, 1])
        driver_result_reduced = trafo.transform(driver_result)

        electronic_energy = ElectronicEnergy.from_raw_integrals(
            np.asarray([[-1.24943841, 0.0], [0.0, -0.547816138]]),
            np.asarray(
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
            ),
        )
        electronic_energy.constants["ActiveSpaceTransformer"] = 0.0
        expected = ElectronicStructureProblem(electronic_energy)
        expected.num_particles = nelec
        expected.num_spatial_orbitals = norb
        dipole_moment = ElectronicDipoleMoment(
            ElectronicIntegrals.from_raw_integrals(np.zeros((2, 2))),
            ElectronicIntegrals.from_raw_integrals(np.zeros((2, 2))),
            ElectronicIntegrals.from_raw_integrals(
                np.asarray([[0.69447435, -1.01418298], [-1.01418298, 0.69447435]]),
            ),
        )
        dipole_moment.constants["ActiveSpaceTransformer"] = (0.0, 0.0, 0.0)
        expected.properties.electronic_dipole_moment = dipole_moment

        self.assertDriverResult(driver_result_reduced, expected)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_numpy_integer(self):
        """Tests that numpy integer objects do not cause issues in `isinstance` checks.

        This is a regression test against the fix applied by
        https://github.com/Qiskit/qiskit-nature/pull/712
        """
        driver = PySCFDriver(basis="631g")
        driver_result = driver.run()
        driver_result.num_spatial_orbitals = np.int64(driver_result.num_spatial_orbitals)
        driver_result.num_particles = (
            np.int64(driver_result.num_alpha),
            np.int64(driver_result.num_beta),
        )

        trafo = ActiveSpaceTransformer(driver_result.num_particles, 2)
        _ = trafo.transform(driver_result)


if __name__ == "__main__":
    unittest.main()
