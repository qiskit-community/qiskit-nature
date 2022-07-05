# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the HDF5 methods."""

import os
import tempfile
from test import QiskitNatureTestCase
from qiskit_nature import QiskitNatureError
from qiskit_nature.hdf5 import load_from_hdf5, save_to_hdf5
from qiskit_nature.second_q.properties import (
    ElectronicStructureDriverResult,
)


class TestHDF5(QiskitNatureTestCase):
    """Tests for the HDF5 methods."""

    def test_load_from_hdf5(self):
        """Test load_from_hdf5."""
        with self.subTest("Normal behavior"):
            driver_result = load_from_hdf5(
                self.get_resource_path(
                    "electronic_structure_driver_result.hdf5",
                    "second_q/properties/resources",
                )
            )
            self.assertTrue(isinstance(driver_result, ElectronicStructureDriverResult))

        with self.subTest("multiple groups"):
            with self.assertRaisesRegex(QiskitNatureError, "more than one HDF5Storable"):
                _ = load_from_hdf5(
                    self.get_resource_path("test_hdf5_error_multiple_groups.hdf5", "resources")
                )

        with self.subTest("missing class"):
            with self.assertRaisesRegex(QiskitNatureError, "faulty object without a '__class__'"):
                _ = load_from_hdf5(
                    self.get_resource_path("test_hdf5_error_missing_class.hdf5", "resources")
                )

        with self.subTest("non native"):
            with self.assertRaisesRegex(QiskitNatureError, "non-native object"):
                _ = load_from_hdf5(
                    self.get_resource_path("test_hdf5_error_non_native.hdf5", "resources")
                )

        with self.subTest("import failure"):
            with self.assertRaisesRegex(QiskitNatureError, "import failure of .+ from .+"):
                _ = load_from_hdf5(
                    self.get_resource_path("test_hdf5_error_import_failure.hdf5", "resources")
                )

        with self.subTest("non protocol"):
            with self.assertRaisesRegex(
                QiskitNatureError, "object of type .+ which is not an HDF5Storable"
            ):
                _ = load_from_hdf5(
                    self.get_resource_path("test_hdf5_error_non_protocol.hdf5", "resources")
                )

    def test_save_to_hdf5(self):
        """Test save_to_hdf5."""
        driver_result = load_from_hdf5(
            self.get_resource_path(
                "electronic_structure_driver_result.hdf5",
                "second_q/properties/resources",
            )
        )

        with self.subTest("Normal behavior"):
            # pylint: disable=consider-using-with
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
            tmp_file.close()
            os.unlink(tmp_file.name)
            save_to_hdf5(driver_result, tmp_file.name)
            try:
                self.assertTrue(os.path.exists(tmp_file.name))
            finally:
                os.unlink(tmp_file.name)

        with self.subTest("FileExistsError"):
            with tempfile.NamedTemporaryFile() as tmp_file:
                with self.assertRaises(FileExistsError):
                    save_to_hdf5(driver_result, tmp_file.name)

        with self.subTest("replace=True"):
            # pylint: disable=consider-using-with
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
            # we need to use delete=False here because on Windows it is not possible to open an
            # existing file a second time:
            # https://docs.python.org/3.9/library/tempfile.html#tempfile.NamedTemporaryFile
            tmp_file.close()
            try:
                save_to_hdf5(driver_result, tmp_file.name, replace=True)
            finally:
                os.unlink(tmp_file.name)
