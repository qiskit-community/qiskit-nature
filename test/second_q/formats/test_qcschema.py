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

import os
import unittest
from tempfile import NamedTemporaryFile

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
        # pylint: disable=consider-using-with
        tmp_file = NamedTemporaryFile(delete=False, suffix=".hdf5")
        tmp_file.close()
        os.unlink(tmp_file.name)
        try:
            with h5py.File(tmp_file.name, "w") as file:
                EXPECTED_WATER_OUTPUT_V3.to_hdf5(file)
            qcs = QCSchema.from_hdf5(tmp_file.name)
            self.assertEqual(qcs, EXPECTED_WATER_OUTPUT_V3)
        finally:
            os.unlink(tmp_file.name)


if __name__ == "__main__":
    unittest.main()
