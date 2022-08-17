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
from test.second_q.drivers.test_driver_methods_gsc import TestDriverMethods
from qiskit_nature.second_q.drivers import FCIDumpDriver
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.exceptions import QiskitNatureError


class TestDriverMethodsFCIDump(TestDriverMethods):
    """Driver Methods FCIDump tests"""

    def test_lih(self):
        """LiH test"""
        driver = FCIDumpDriver(
            self.get_resource_path("test_driver_fcidump_lih.fcidump", "second_q/drivers/fcidumpd")
        )
        result = self._run_driver(driver)
        self._assert_energy(result, "lih")

    def test_oh(self):
        """OH test"""
        driver = FCIDumpDriver(
            self.get_resource_path("test_driver_fcidump_oh.fcidump", "second_q/drivers/fcidumpd")
        )
        result = self._run_driver(driver)
        self._assert_energy(result, "oh")

    def test_lih_freeze_core(self):
        """LiH freeze core test"""
        with self.assertRaises(QiskitNatureError) as ctx:
            driver = FCIDumpDriver(
                self.get_resource_path(
                    "test_driver_fcidump_lih.fcidump", "second_q/drivers/fcidumpd"
                )
            )
            result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
            self._assert_energy(result, "lih")
        msg = (
            "'The provided `GroupedElectronicProperty` does not contain an `ElectronicBasisTransform`"
            " property, which is required by this transformer!'"
        )
        self.assertEqual(msg, str(ctx.exception))

    def test_oh_freeze_core(self):
        """OH freeze core test"""
        with self.assertRaises(QiskitNatureError) as ctx:
            driver = FCIDumpDriver(
                self.get_resource_path(
                    "test_driver_fcidump_oh.fcidump", "second_q/drivers/fcidumpd"
                )
            )
            result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
            self._assert_energy(result, "oh")
        msg = (
            "'The provided `GroupedElectronicProperty` does not contain an `ElectronicBasisTransform`"
            " property, which is required by this transformer!'"
        )
        self.assertEqual(msg, str(ctx.exception))

    @unittest.skip("Skip until FreezeCoreTransformer supports pure MO cases.")
    def test_lih_with_atoms(self):
        """LiH with num_atoms test"""
        driver = FCIDumpDriver(
            self.get_resource_path("test_driver_fcidump_lih.fcidump", "second_q/drivers/fcidumpd"),
            # atoms=["Li", "H"],
        )
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy(result, "lih")

    @unittest.skip("Skip until FreezeCoreTransformer supports pure MO cases.")
    def test_oh_with_atoms(self):
        """OH with num_atoms test"""
        driver = FCIDumpDriver(
            self.get_resource_path("test_driver_fcidump_oh.fcidump", "second_q/drivers/fcidumpd"),
            # atoms=["O", "H"],
        )
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy(result, "oh")


class TestFCIDumpDriverDriverResult(QiskitNatureTestCase):
    """DriverResult FCIDumpDriver tests."""

    def test_driver_result_log(self):
        """Test DriverResult log function."""
        driver_result = FCIDumpDriver(
            self.get_resource_path("test_driver_fcidump_h2.fcidump", "second_q/drivers/fcidumpd")
        ).run()
        with self.assertLogs("qiskit_nature", level="DEBUG") as _:
            driver_result.log()

    def test_driver_result_log_with_atoms(self):
        """Test DriverResult log function."""
        driver_result = FCIDumpDriver(
            self.get_resource_path("test_driver_fcidump_h2.fcidump", "second_q/drivers/fcidumpd"),
            # atoms=["H", "H"],
        ).run()
        with self.assertLogs("qiskit_nature", level="DEBUG") as _:
            driver_result.log()


if __name__ == "__main__":
    unittest.main()
