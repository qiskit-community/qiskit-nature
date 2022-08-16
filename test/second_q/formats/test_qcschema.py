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

"""Test the QCSchema implementation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from test import QiskitNatureTestCase
from test.second_q.formats.qcschema.he2_energy_VV10_input import EXPECTED as EXPECTED_HE2_INPUT
from test.second_q.formats.qcschema.he2_energy_VV10_output import EXPECTED as EXPECTED_HE2_OUTPUT
from test.second_q.formats.qcschema.water_output import EXPECTED as EXPECTED_WATER_OUTPUT_V1
from test.second_q.formats.qcschema.water_output_v3 import EXPECTED as EXPECTED_WATER_OUTPUT_V3

from ddt import ddt, data, unpack

import h5py

from qiskit_nature.second_q.formats.qcschema import QCSchema, QCSchemaInput


@ddt
class TestQCSchemaJSON(QiskitNatureTestCase):
    """Tests the QCSchema JSON capabilities."""

    @unpack
    @data(
        (QCSchemaInput, "he2_energy_VV10_input.json", EXPECTED_HE2_INPUT),
        (QCSchema, "he2_energy_VV10_output.json", EXPECTED_HE2_OUTPUT),
        (QCSchema, "water_output.json", EXPECTED_WATER_OUTPUT_V1),
        (QCSchema, "water_output_v3.json", EXPECTED_WATER_OUTPUT_V3),
    )
    def test_from_json(self, qcschema, filename, expected):
        """Tests the from_json parsing."""
        file = self.get_resource_path(filename, "second_q/formats/qcschema")
        qcs = qcschema.from_json(file)
        self.assertEqual(qcs, expected)

    def test_to_json(self):
        """Tests the to_json dumping.

        Note: this test relies on the `from_json` parsing to work correctly.
        """
        json_string = EXPECTED_WATER_OUTPUT_V3.to_json()
        qcs = QCSchema.from_json(json_string)
        self.assertEqual(qcs, EXPECTED_WATER_OUTPUT_V3)


@unittest.skip("Skip until we have settled on the HDF5 specification.")
@ddt
class TestQCSchemaHDF5(QiskitNatureTestCase):
    """Tests the QCSchema HDF5 capabilities."""

    @unpack
    @data(
        (QCSchemaInput, "he2_energy_VV10_input.hdf5", EXPECTED_HE2_INPUT),
        (QCSchema, "he2_energy_VV10_output.hdf5", EXPECTED_HE2_OUTPUT),
        (QCSchema, "water_output.hdf5", EXPECTED_WATER_OUTPUT_V1),
        (QCSchema, "water_output_v3.hdf5", EXPECTED_WATER_OUTPUT_V3),
    )
    def test_from_hdf5(self, qcschema, filename, expected):
        """Tests the from_hdf5 parsing."""
        file = self.get_resource_path(filename, "second_q/formats/qcschema")
        qcs = qcschema.from_hdf5(file)
        self.assertEqual(qcs, expected)

    def test_to_hdf5(self):
        """Tests the to_hdf5 dumping.

        Note: this test relies on the `from_hdf5` parsing to work correctly.
        """
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "tmp.hdf5"
            with h5py.File(file_path, "w") as file:
                EXPECTED_WATER_OUTPUT_V3.to_hdf5(file)
            qcs = QCSchema.from_hdf5(file_path)
            self.assertEqual(qcs, EXPECTED_WATER_OUTPUT_V3)


class TestQCSchemaLegacy(QiskitNatureTestCase):
    """Tests the QCSchema.from_legacy_hdf5 method."""

    def test_legacy_from_hdf5(self):
        """Tests the legacy_from_hdf5 method."""
        with self.subTest("ElectronicStructureDriverResult"):
            qcschema = QCSchema.from_legacy_hdf5(
                self.get_resource_path(
                    "electronic_structure_driver_result.hdf5",
                    "properties/second_quantization/electronic/resources",
                )
            )

            expected = QCSchema.from_json(
                self.get_resource_path(
                    "legacy_electronic_structure_driver_result.json",
                    "second_q/formats/qcschema",
                )
            )

            self.assertEqual(qcschema, expected)

        with self.subTest("Error on non-electronic case"):
            with self.assertRaises(ValueError):
                qcschema = QCSchema.from_legacy_hdf5(
                    self.get_resource_path(
                        "vibrational_structure_driver_result.hdf5",
                        "properties/second_quantization/vibrational/resources",
                    )
                )


if __name__ == "__main__":
    unittest.main()
