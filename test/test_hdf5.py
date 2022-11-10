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

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from test import QiskitNatureDeprecatedTestCase
from qiskit_nature import QiskitNatureError
from qiskit_nature.hdf5 import load_from_hdf5, save_to_hdf5
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicStructureDriverResult,
)


class TestHDF5(QiskitNatureDeprecatedTestCase):
    """Tests for the HDF5 methods."""

    def test_load_from_hdf5(self):
        """Test load_from_hdf5."""
        with self.subTest("Normal behavior"):
            driver_result = load_from_hdf5(
                self.get_resource_path(
                    "electronic_structure_driver_result.hdf5",
                    "properties/second_quantization/electronic/resources",
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
                "properties/second_quantization/electronic/resources",
            )
        )

        with self.subTest("Normal behavior"):
            with TemporaryDirectory() as tmp_dir:
                tmp_file = Path(tmp_dir) / "tmp.hdf5"
                save_to_hdf5(driver_result, tmp_file)
                self.assertTrue(tmp_file.exists())

        with self.subTest("FileExistsError"):
            with NamedTemporaryFile() as tmp_file:
                with self.assertRaises(FileExistsError):
                    save_to_hdf5(driver_result, tmp_file.name)

        with self.subTest("replace=True"):
            with TemporaryDirectory() as tmp_dir:
                tmp_file = Path(tmp_dir) / "tmp.hdf5"
                tmp_file.touch()
                self.assertTrue(tmp_file.exists())
                save_to_hdf5(driver_result, tmp_file, replace=True)
