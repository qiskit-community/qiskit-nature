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

from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import unittest
import warnings

from test import QiskitNatureDeprecatedTestCase
from test.drivers.second_quantization.test_driver import TestDriver
from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.drivers import QMolecule
from qiskit_nature.properties.second_quantization.electronic import ElectronicStructureDriverResult


class TestDriverHDF5(QiskitNatureDeprecatedTestCase, TestDriver):
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
            with TemporaryDirectory() as tmp_dir:
                tmp_file = Path(tmp_dir) / "tmp.hdf5"
                shutil.copy(legacy_file_path, tmp_file)
                driver = HDF5Driver(tmp_file)
                # replacing file won't trigger deprecation on run
                driver.convert(replace=True)

        msg_mol_ref = (
            "The HDF5Driver.run with legacy HDF5 file method is deprecated as of version 0.4.0 "
            "and will be removed no sooner than 3 months after the release "
            ". Your HDF5 file contains the legacy QMolecule object! You should "
            "consider converting it to the new property framework. See also HDF5Driver.convert."
        )
        with self.subTest("replace=False"):
            with TemporaryDirectory() as tmp_dir:
                tmp_file = Path(tmp_dir) / "tmp.hdf5"
                new_file_name = Path(tmp_dir) / "tmp_new.hdf5"
                shutil.copy(legacy_file_path, tmp_file)
                driver = HDF5Driver(tmp_file)
                # not replacing file will trigger deprecation on run
                driver.convert(replace=False)
                with warnings.catch_warnings(record=True) as c_m:
                    warnings.simplefilter("always")
                    driver.run()
                    self.assertEqual(str(c_m[0].message), msg_mol_ref)

                # using new file won't trigger deprecation
                HDF5Driver(new_file_name).run()


class TestDriverHDF5Legacy(QiskitNatureDeprecatedTestCase, TestDriver):
    """HDF5 Driver legacy file-support tests."""

    def setUp(self):
        super().setUp()
        hdf5_file = self.get_resource_path(
            "test_driver_hdf5_legacy.hdf5", "drivers/second_quantization/hdf5d"
        )
        # Using QMolecule directly here to avoid the deprecation on HDF5Driver.run method
        # to be triggered and let it be handled on the method test_convert
        # Those deprecation messages are shown only once and this one could prevent
        # the test_convert one to show if called first.
        molecule = QMolecule(hdf5_file)
        molecule.load()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.driver_result = ElectronicStructureDriverResult.from_legacy_driver_result(molecule)
        warnings.filterwarnings("default", category=DeprecationWarning)


if __name__ == "__main__":
    unittest.main()
