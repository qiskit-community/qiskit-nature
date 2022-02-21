# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver HDF5 """

import os
import pathlib
import shutil
import tempfile
import unittest

from test import QiskitNatureTestCase
from test.drivers.second_quantization.test_driver import TestDriver
from qiskit_nature.drivers.second_quantization import HDF5Driver


class TestDriverHDF5(QiskitNatureTestCase, TestDriver):
    """HDF5 Driver tests."""

    def setUp(self):
        super().setUp()
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "test_driver_hdf5.hdf5", "drivers/second_quantization/hdf5d"
            )
        )
        self.driver_result = driver.run()

    def test_convert(self):
        """Test the legacy-conversion method."""
        legacy_file_path = self.get_resource_path(
            "test_driver_hdf5_legacy.hdf5", "drivers/second_quantization/hdf5d"
        )

        with self.subTest("replace=True"):
            # pylint: disable=consider-using-with
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
            tmp_file.close()
            os.unlink(tmp_file.name)
            shutil.copy(legacy_file_path, tmp_file.name)
            try:
                driver = HDF5Driver(tmp_file.name)
                driver.convert(replace=True)
                with self.assertRaises(AssertionError):
                    # NOTE: we can use assertNoLogs once Python 3.10 is the default
                    with self.assertLogs(
                        logger="qiskit_nature.drivers.second_quantization.hdf5d.hdf5driver",
                        level="WARNING",
                    ):
                        driver.run()
            finally:
                os.unlink(tmp_file.name)

        with self.subTest("replace=False"):
            # pylint: disable=consider-using-with
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
            tmp_file.close()
            new_file_name = pathlib.Path(tmp_file.name).with_stem(
                str(pathlib.Path(tmp_file.name).stem) + "_new"
            )
            os.unlink(tmp_file.name)
            shutil.copy(legacy_file_path, tmp_file.name)
            try:
                driver = HDF5Driver(tmp_file.name)
                driver.convert(replace=False)
                with self.assertLogs(
                    logger="qiskit_nature.drivers.second_quantization.hdf5d.hdf5driver",
                    level="WARNING",
                ):
                    driver.run()
                with self.assertRaises(AssertionError):
                    # NOTE: we can use assertNoLogs once Python 3.10 is the default
                    with self.assertLogs(
                        logger="qiskit_nature.drivers.second_quantization.hdf5d.hdf5driver",
                        level="WARNING",
                    ):
                        HDF5Driver(new_file_name).run()
            finally:
                os.unlink(tmp_file.name)
                os.unlink(new_file_name)


class TestDriverHDF5Legacy(QiskitNatureTestCase, TestDriver):
    """HDF5 Driver legacy file-support tests."""

    def setUp(self):
        super().setUp()
        driver = HDF5Driver(
            hdf5_input=self.get_resource_path(
                "test_driver_hdf5_legacy.hdf5", "drivers/second_quantization/hdf5d"
            )
        )
        self.driver_result = driver.run()


if __name__ == "__main__":
    unittest.main()
