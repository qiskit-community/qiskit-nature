# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver Methods FCIDump """

import unittest

from test import QiskitNatureTestCase
from test.drivers.second_quantization.test_driver_methods_gsc import TestDriverMethods
from qiskit_nature.drivers.second_quantization import FCIDumpDriver
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer


class TestDriverMethodsFCIDump(TestDriverMethods):
    """Driver Methods FCIDump tests"""

    def test_lih(self):
        """LiH test"""
        driver = FCIDumpDriver(
            self.get_resource_path(
                "test_driver_fcidump_lih.fcidump", "drivers/second_quantization/fcidumpd"
            )
        )
        result = self._run_driver(driver)
        self._assert_energy(result, "lih")

    def test_oh(self):
        """OH test"""
        driver = FCIDumpDriver(
            self.get_resource_path(
                "test_driver_fcidump_oh.fcidump", "drivers/second_quantization/fcidumpd"
            )
        )
        result = self._run_driver(driver)
        self._assert_energy(result, "oh")

    @unittest.skip("Skip until FreezeCoreTransformer supports pure MO cases.")
    def test_lih_freeze_core(self):
        """LiH freeze core test"""
        with self.assertLogs("qiskit_nature", level="WARNING") as log:
            driver = FCIDumpDriver(
                self.get_resource_path(
                    "test_driver_fcidump_lih.fcidump", "drivers/second_quantization/fcidumpd"
                )
            )
            result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
            self._assert_energy(result, "lih")
        warning = (
            "WARNING:qiskit_nature.drivers.second_quantization.qmolecule:Missing molecule !"
            "information! Returning empty core orbital list."
        )
        self.assertIn(warning, log.output)

    @unittest.skip("Skip until FreezeCoreTransformer supports pure MO cases.")
    def test_oh_freeze_core(self):
        """OH freeze core test"""
        with self.assertLogs("qiskit_nature", level="WARNING") as log:
            driver = FCIDumpDriver(
                self.get_resource_path(
                    "test_driver_fcidump_oh.fcidump", "drivers/second_quantization/fcidumpd"
                )
            )
            result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
            self._assert_energy(result, "oh")
        warning = (
            "WARNING:qiskit_nature.drivers.second_quantization.qmolecule:Missing molecule "
            "information! Returning empty core orbital list."
        )
        self.assertIn(warning, log.output)

    @unittest.skip("Skip until FreezeCoreTransformer supports pure MO cases.")
    def test_lih_with_atoms(self):
        """LiH with num_atoms test"""
        driver = FCIDumpDriver(
            self.get_resource_path(
                "test_driver_fcidump_lih.fcidump", "drivers/second_quantization/fcidumpd"
            ),
            # atoms=["Li", "H"],
        )
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy(result, "lih")

    @unittest.skip("Skip until FreezeCoreTransformer supports pure MO cases.")
    def test_oh_with_atoms(self):
        """OH with num_atoms test"""
        driver = FCIDumpDriver(
            self.get_resource_path(
                "test_driver_fcidump_oh.fcidump", "drivers/second_quantization/fcidumpd"
            ),
            # atoms=["O", "H"],
        )
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy(result, "oh")


class TestFCIDumpDriverDriverResult(QiskitNatureTestCase):
    """DriverResult FCIDumpDriver tests."""

    def test_driver_result_log(self):
        """Test DriverResult log function."""
        driver_result = FCIDumpDriver(
            self.get_resource_path(
                "test_driver_fcidump_h2.fcidump", "drivers/second_quantization/fcidumpd"
            )
        ).run()
        with self.assertLogs("qiskit_nature", level="DEBUG") as _:
            driver_result.log()

    def test_driver_result_log_with_atoms(self):
        """Test DriverResult log function."""
        driver_result = FCIDumpDriver(
            self.get_resource_path(
                "test_driver_fcidump_h2.fcidump", "drivers/second_quantization/fcidumpd"
            ),
            # atoms=["H", "H"],
        ).run()
        with self.assertLogs("qiskit_nature", level="DEBUG") as _:
            driver_result.log()


if __name__ == "__main__":
    unittest.main()
