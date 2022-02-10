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

"""
Code inside the test is the chemistry sample from the readme.
If this test fails and code changes are needed here to resolve
the issue then ensure changes are made to readme too.
"""

import unittest

import contextlib
import io
from pathlib import Path
import re
from test import QiskitNatureTestCase, requires_extra_library


class TestReadmeSample(QiskitNatureTestCase):
    """Test sample code from readme"""

    @requires_extra_library
    def setUp(self):
        super().setUp()
        # pylint: disable=import-outside-toplevel
        from qiskit_nature.drivers.second_quantization import PySCFDriver

        PySCFDriver(atom="Li .0 .0 .0; H .0 .0 1.6")

        try:
            # pylint: disable=import-outside-toplevel
            # pylint: disable=unused-import
            from qiskit import Aer

            _ = Aer.get_backend("aer_simulator_statevector")
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")
            return

    def test_readme_sample(self):
        """readme sample test"""
        # pylint: disable=exec-used

        readme_name = "README.md"
        readme_path = Path(__file__).parent.parent.joinpath(readme_name)
        if not readme_path.exists() or not readme_path.is_file():
            self.fail(msg=f"{readme_name} not found at {readme_path}")
            return

        # gets the first matched code sample
        # assumes one code sample to test per readme
        readme_sample = None
        with open(readme_path, encoding="UTF-8") as readme_file:
            match_sample = re.search(
                "```python.*```",
                readme_file.read(),
                flags=re.S,
            )
            if match_sample:
                # gets the matched string stripping the markdown code block
                readme_sample = match_sample.group(0)[9:-3]

        if readme_sample is None:
            self.skipTest(f"No sample found inside {readme_name}.")
            return

        with contextlib.redirect_stdout(io.StringIO()) as out:
            try:
                exec(readme_sample)
            except Exception as ex:  # pylint: disable=broad-except
                self.fail(str(ex))
                return

        texts = out.getvalue().split("\n")
        self.assertAlmostEqual(float(texts[0]), -1.8572750301938803, places=6)


if __name__ == "__main__":
    unittest.main()
