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

from test import QiskitNatureTestCase
from test.second_q.formats.qcschema.he2_energy_VV10_input import EXPECTED as EXPECTED_HE2_INPUT
from test.second_q.formats.qcschema.he2_energy_VV10_output import EXPECTED as EXPECTED_HE2_OUTPUT
from test.second_q.formats.qcschema.water_output import EXPECTED as EXPECTED_WATER_OUTPUT_V1
from test.second_q.formats.qcschema.water_output_v3 import EXPECTED as EXPECTED_WATER_OUTPUT_V3

from ddt import ddt, data, unpack

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


if __name__ == "__main__":
    unittest.main()
