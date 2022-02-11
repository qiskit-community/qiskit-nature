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
from qiskit_nature.hdf5 import load_from_hdf5, save_to_hdf5
from qiskit_nature.properties.second_quantization.electronic import ElectronicStructureDriverResult


class TestHDF5(QiskitNatureTestCase):
    """Tests for the HDF5 methods."""

    def test_load_from_hdf5(self):
        """Test load_from_hdf5."""
        driver_result = load_from_hdf5(
            self.get_resource_path(
                "electronic_structure_driver_result.hdf5",
                "properties/second_quantization/electronic/resources",
            )
        )
        self.assertTrue(isinstance(driver_result, ElectronicStructureDriverResult))

    def test_save_to_hdf5(self):
        """Test save_to_hdf5."""
        driver_result = load_from_hdf5(
            self.get_resource_path(
                "electronic_structure_driver_result.hdf5",
                "properties/second_quantization/electronic/resources",
            )
        )

        with self.subTest("Normal behavior"):
            # pylint: disable=consider-using-with
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
            tmp_file.close()
            os.unlink(tmp_file.name)
            save_to_hdf5(driver_result, tmp_file.name)
            self.assertTrue(os.path.exists(tmp_file.name))
            os.unlink(tmp_file.name)

        with self.subTest("FileExistsError"):
            with tempfile.NamedTemporaryFile() as tmp_file:
                with self.assertRaises(FileExistsError):
                    save_to_hdf5(driver_result, tmp_file.name)

        with self.subTest("replace=True"):
            with tempfile.NamedTemporaryFile() as tmp_file:
                save_to_hdf5(driver_result, tmp_file.name, replace=True)
