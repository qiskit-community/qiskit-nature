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

"""Tests for the FreezeCoreTransformer."""

import unittest

from test import QiskitNatureTestCase
from ddt import ddt, idata

from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.properties.second_quantization.electronic import ElectronicStructureDriverResult
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer


# With Python 3.6 this false positive is being raised for the ElectronicStructureDriverResult
# pylint: disable=abstract-class-instantiated
@ddt
class TestFreezeCoreTransformer(QiskitNatureTestCase):
    """FreezeCoreTransformer tests."""

    # pylint: disable=import-outside-toplevel
    from test.transformers.second_quantization.electronic.test_active_space_transformer import (
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
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "H2_sto3g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        q_molecule = driver.run()
        driver_result = ElectronicStructureDriverResult.from_legacy_driver_result(q_molecule)

        # The references which we compare too were produced by the `ActiveSpaceTransformer` and,
        # thus, the key here needs to stay the same as in that test case.
        driver_result.get_property("ElectronicEnergy")._shift["ActiveSpaceTransformer"] = 0.0
        for prop in iter(driver_result.get_property("ElectronicDipoleMoment")):
            prop._shift["ActiveSpaceTransformer"] = 0.0

        trafo = FreezeCoreTransformer(**kwargs)
        driver_result_reduced = trafo.transform(driver_result)

        self.assertDriverResult(
            driver_result_reduced, driver_result, dict_key="FreezeCoreTransformer"
        )

    def test_freeze_core(self):
        """Test the `freeze_core` convenience argument."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "LiH_sto3g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        q_molecule = driver.run()
        driver_result = ElectronicStructureDriverResult.from_legacy_driver_result(q_molecule)

        trafo = FreezeCoreTransformer(freeze_core=True)
        driver_result_reduced = trafo.transform(driver_result)

        expected_qmol = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "LiH_sto3g_reduced.hdf5", "transformers/second_quantization/electronic"
            )
        ).run()
        expected_qmol.num_molecular_orbitals = 4
        expected = ElectronicStructureDriverResult.from_legacy_driver_result(expected_qmol)

        self.assertDriverResult(driver_result_reduced, expected, dict_key="FreezeCoreTransformer")

    def test_freeze_core_with_remove_orbitals(self):
        """Test the `freeze_core` convenience argument in combination with `remove_orbitals`."""
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "BeH_sto3g.hdf5", "transformers/second_quantization/electronic"
            )
        )
        q_molecule = driver.run()
        driver_result = ElectronicStructureDriverResult.from_legacy_driver_result(q_molecule)

        trafo = FreezeCoreTransformer(freeze_core=True, remove_orbitals=[4, 5])
        driver_result_reduced = trafo.transform(driver_result)

        expected_qmol = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "BeH_sto3g_reduced.hdf5", "transformers/second_quantization/electronic"
            )
        ).run()
        expected_qmol.num_molecular_orbitals = 3
        expected = ElectronicStructureDriverResult.from_legacy_driver_result(expected_qmol)

        self.assertDriverResult(driver_result_reduced, expected, dict_key="FreezeCoreTransformer")


if __name__ == "__main__":
    unittest.main()
