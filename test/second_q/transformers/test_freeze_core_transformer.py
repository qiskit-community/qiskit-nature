# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the FreezeCoreTransformer."""

import unittest

from test import QiskitNatureTestCase
from test.second_q.utils import get_expected_two_body_ints

from ddt import ddt, idata
import numpy as np

import qiskit_nature.optionals as _optionals
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.qcschema_translator import qcschema_to_problem
from qiskit_nature.second_q.operators.tensor_ordering import to_chemist_ordering
from qiskit_nature.second_q.transformers import FreezeCoreTransformer


@ddt
class TestFreezeCoreTransformer(QiskitNatureTestCase):
    """FreezeCoreTransformer tests."""

    from test.second_q.transformers.test_active_space_transformer import (
        TestActiveSpaceTransformer,
    )

    assertDriverResult = TestActiveSpaceTransformer.assertDriverResult
    assertElectronicEnergy = TestActiveSpaceTransformer.assertElectronicEnergy

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    @idata(
        [
            {"freeze_core": True},
        ]
    )
    def test_full_active_space(self, kwargs):
        """Test that transformer has no effect when all orbitals are active."""
        driver = PySCFDriver()
        driver_result = driver.run()

        driver_result.hamiltonian.constants["FreezeCoreTransformer"] = 0.0
        driver_result.properties.electronic_dipole_moment.constants["FreezeCoreTransformer"] = (
            0.0,
            0.0,
            0.0,
        )

        trafo = FreezeCoreTransformer(**kwargs)
        driver_result_reduced = trafo.transform(driver_result)

        self.assertDriverResult(driver_result_reduced, driver_result)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_freeze_core(self):
        """Test the `freeze_core` convenience argument."""
        driver = PySCFDriver(atom="Li 0 0 0; H 0 0 1.6")
        driver_result = driver.run()

        trafo = FreezeCoreTransformer(freeze_core=True)
        driver_result_reduced = trafo.transform(driver_result)

        expected = qcschema_to_problem(
            QCSchema.from_json(
                self.get_resource_path("LiH_sto3g_reduced.json", "second_q/transformers/resources")
            ),
            include_dipole=False,
        )
        # add energy shift, which currently cannot be stored in the QCSchema
        expected.hamiltonian.constants["FreezeCoreTransformer"] = -7.796219568771229

        self.assertDriverResult(driver_result_reduced, expected)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_freeze_core_with_remove_orbitals(self):
        """Test the `freeze_core` convenience argument in combination with `remove_orbitals`."""
        driver = PySCFDriver(atom="Be 0 0 0; H 0 0 1.3", basis="sto3g", spin=1)
        driver_result = driver.run()

        trafo = FreezeCoreTransformer(freeze_core=True, remove_orbitals=[4, 5])
        driver_result_reduced = trafo.transform(driver_result)

        expected = qcschema_to_problem(
            QCSchema.from_json(
                self.get_resource_path("BeH_sto3g_reduced.json", "second_q/transformers/resources")
            ),
            include_dipole=False,
        )
        # add energy shift, which currently cannot be stored in the QCSchema
        expected.hamiltonian.constants["FreezeCoreTransformer"] = -14.253802923103054

        self.assertDriverResult(driver_result_reduced, expected)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_no_freeze_core(self):
        """Test the disabled `freeze_core` convenience argument.

        Regression test against https://github.com/Qiskit/qiskit-nature/issues/652
        """
        driver = PySCFDriver(atom="Li 0 0 0; H 0 0 1.6")
        driver_result = driver.run()

        trafo = FreezeCoreTransformer(freeze_core=False)
        driver_result_reduced = trafo.transform(driver_result)

        electronic_energy = driver_result_reduced.hamiltonian
        electronic_energy_exp = driver_result.hamiltonian
        with self.subTest("MO 1-electron integrals"):
            np.testing.assert_array_almost_equal(
                np.abs(electronic_energy.electronic_integrals.second_q_coeffs()["+-"]),
                np.abs(electronic_energy_exp.electronic_integrals.second_q_coeffs()["+-"]),
            )
        with self.subTest("MO 2-electron integrals"):
            actual_ints = electronic_energy.electronic_integrals.second_q_coeffs()["++--"]
            expected_ints = get_expected_two_body_ints(
                actual_ints,
                to_chemist_ordering(
                    electronic_energy_exp.electronic_integrals.second_q_coeffs()["++--"]
                ),
            )
            np.testing.assert_array_almost_equal(
                np.abs(actual_ints),
                np.abs(expected_ints),
            )
        with self.subTest("Inactive energy"):
            self.assertAlmostEqual(electronic_energy.constants["FreezeCoreTransformer"], 0.0)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_freeze_core_with_charge(self):
        """Test the transformer behavior with a charge present."""
        driver = PySCFDriver(atom="Be 0 0 0", charge=1, spin=1)
        driver_result = driver.run()
        self.assertEqual(driver_result.num_alpha, 2)
        self.assertEqual(driver_result.num_beta, 1)

        trafo = FreezeCoreTransformer(freeze_core=True)
        driver_result_reduced = trafo.transform(driver_result)

        self.assertEqual(driver_result_reduced.num_alpha, 1)
        self.assertEqual(driver_result_reduced.num_beta, 0)

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_standalone_usage(self):
        """Test usage on a standalone Hamiltonian."""
        driver = PySCFDriver(atom="Li 0 0 0; H 0 0 1.6")
        problem = driver.run()

        trafo = FreezeCoreTransformer()

        with self.subTest("prepare_active_space not called yet"):
            with self.assertRaises(QiskitNatureError):
                reduced_hamiltonian = trafo.transform_hamiltonian(problem.hamiltonian)

        trafo.prepare_active_space(problem.molecule, problem.num_spatial_orbitals)

        reduced_hamiltonian = trafo.transform_hamiltonian(problem.hamiltonian)

        expected = qcschema_to_problem(
            QCSchema.from_json(
                self.get_resource_path("LiH_sto3g_reduced.json", "second_q/transformers/resources")
            ),
            include_dipole=False,
        )
        # add energy shift, which currently cannot be stored in the QCSchema
        expected.hamiltonian.constants["FreezeCoreTransformer"] = -7.796219568771229

        self.assertElectronicEnergy(reduced_hamiltonian, expected.hamiltonian)


if __name__ == "__main__":
    unittest.main()
