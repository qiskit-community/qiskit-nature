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

"""Tests for the FreezeCoreTransformer."""

import unittest

from test import QiskitNatureTestCase
from ddt import ddt, idata
import numpy as np

from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.qcschema_translator import qcschema_to_problem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.properties.bases import ElectronicBasis


@ddt
class TestFreezeCoreTransformer(QiskitNatureTestCase):
    """FreezeCoreTransformer tests."""

    # pylint: disable=import-outside-toplevel
    from test.second_q.transformers.test_active_space_transformer import (
        TestActiveSpaceTransformer,
    )

    assertDriverResult = TestActiveSpaceTransformer.assertDriverResult

    @idata(
        [
            {"freeze_core": True},
        ]
    )
    def test_full_active_space(self, kwargs):
        """Test that transformer has no effect when all orbitals are active."""
        qcschema = QCSchema.from_legacy_hdf5(
            self.get_resource_path("H2_sto3g.hdf5", "transformers/second_quantization/electronic")
        )
        driver_result = qcschema_to_problem(qcschema)

        # The references which we compare too were produced by the `ActiveSpaceTransformer` and,
        # thus, the key here needs to stay the same as in that test case.
        driver_result.hamiltonian._shift["ActiveSpaceTransformer"] = 0.0
        for prop in driver_result.properties.electronic_dipole_moment._dipole_axes.values():
            prop._shift["ActiveSpaceTransformer"] = 0.0

        trafo = FreezeCoreTransformer(**kwargs)
        driver_result_reduced = trafo.transform(driver_result)

        self.assertDriverResult(
            driver_result_reduced, driver_result, dict_key="FreezeCoreTransformer"
        )

    def test_freeze_core(self):
        """Test the `freeze_core` convenience argument."""
        qcschema = QCSchema.from_legacy_hdf5(
            self.get_resource_path("LiH_sto3g.hdf5", "transformers/second_quantization/electronic")
        )
        driver_result = qcschema_to_problem(qcschema)

        trafo = FreezeCoreTransformer(freeze_core=True)
        driver_result_reduced = trafo.transform(driver_result)

        qcschema_exp = QCSchema.from_legacy_hdf5(
            self.get_resource_path(
                "LiH_sto3g_reduced.hdf5", "transformers/second_quantization/electronic"
            )
        )
        expected = qcschema_to_problem(qcschema_exp)

        self.assertDriverResult(driver_result_reduced, expected, dict_key="FreezeCoreTransformer")

    def test_freeze_core_with_remove_orbitals(self):
        """Test the `freeze_core` convenience argument in combination with `remove_orbitals`."""
        qcschema = QCSchema.from_legacy_hdf5(
            self.get_resource_path("BeH_sto3g.hdf5", "transformers/second_quantization/electronic")
        )
        driver_result = qcschema_to_problem(qcschema)

        trafo = FreezeCoreTransformer(freeze_core=True, remove_orbitals=[4, 5])
        driver_result_reduced = trafo.transform(driver_result)

        qcschema_exp = QCSchema.from_legacy_hdf5(
            self.get_resource_path(
                "BeH_sto3g_reduced.hdf5", "transformers/second_quantization/electronic"
            )
        )
        expected = qcschema_to_problem(qcschema_exp)
        expected.properties.particle_number.num_spin_orbitals = 6

        self.assertDriverResult(driver_result_reduced, expected, dict_key="FreezeCoreTransformer")

    def test_no_freeze_core(self):
        """Test the disabled `freeze_core` convenience argument.

        Regression test against https://github.com/Qiskit/qiskit-nature/issues/652
        """
        qcschema = QCSchema.from_legacy_hdf5(
            self.get_resource_path("LiH_sto3g.hdf5", "transformers/second_quantization/electronic")
        )
        driver_result = qcschema_to_problem(qcschema)

        trafo = FreezeCoreTransformer(freeze_core=False)
        driver_result_reduced = trafo.transform(driver_result)

        electronic_energy = driver_result_reduced.hamiltonian
        electronic_energy_exp = driver_result.hamiltonian
        with self.subTest("MO 1-electron integrals"):
            np.testing.assert_array_almost_equal(
                electronic_energy.get_electronic_integral(ElectronicBasis.MO, 1).to_spin(),
                electronic_energy_exp.get_electronic_integral(ElectronicBasis.MO, 1).to_spin(),
            )
        with self.subTest("MO 2-electron integrals"):
            np.testing.assert_array_almost_equal(
                electronic_energy.get_electronic_integral(ElectronicBasis.MO, 2).to_spin(),
                electronic_energy_exp.get_electronic_integral(ElectronicBasis.MO, 2).to_spin(),
            )
        with self.subTest("Inactive energy"):
            self.assertAlmostEqual(electronic_energy._shift["FreezeCoreTransformer"], 0.0)


if __name__ == "__main__":
    unittest.main()
