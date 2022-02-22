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

"""
Test Nature logging methods
"""
from typing import List, Dict, Set
import sys
import unittest
import logging
import tempfile
import contextlib
import io
import os
from test import QiskitNatureTestCase
from qiskit_nature import logging as nature_logging


class TestHandler(logging.StreamHandler):
    """Unit Test Handler"""

    def __init__(self):
        """
        Initialize the handler.
        """
        super().__init__(sys.stdout)
        self.records: List[logging.LogRecord] = []

    def emit(self, record) -> None:
        """handle record"""
        self.records.append(record)


class TestLogging(QiskitNatureTestCase):
    """Test logging"""

    def setUp(self):
        super().setUp()
        self._test_handler = TestHandler()
        self._logging_dict = {"qiskit_nature": logging.DEBUG, "qiskit": logging.DEBUG}
        self._old_logging_dict = nature_logging.get_levels_for_names(self._logging_dict.keys())

    def _set_logging(self, use_default_handler: bool):
        nature_logging.set_levels_for_names(
            self._logging_dict, add_default_handler=use_default_handler
        )
        if not use_default_handler:
            for name in self._logging_dict:
                nature_logging.add_handler(name, handler=self._test_handler)

    def tearDown(self) -> None:
        super().tearDown()
        for name in self._logging_dict:
            nature_logging.remove_handler(name, handler=self._test_handler)
        nature_logging.set_levels_for_names(self._old_logging_dict, add_default_handler=False)
        nature_logging.remove_default_handler(self._logging_dict.keys())

    def _validate_records(self, records):
        name_levels: Dict[str, Set[int]] = {}
        for record in records:
            names = record.name.split(".")
            if names[0] in name_levels:
                name_levels[names[0]].add(record.levelno)
            else:
                name_levels[names[0]] = set()
        self.assertCountEqual(
            name_levels.keys(),
            self._logging_dict.keys(),
            msg="Handled logging modules different from reference",
        )
        for name, ref_level in self._logging_dict.items():
            for level in name_levels[name]:
                self.assertGreaterEqual(
                    level,
                    ref_level,
                    msg=f"{name}: logging level {level} < reference level {ref_level} ",
                )

    def test_logging_to_handler(self):
        """logging test"""
        self._set_logging(False)
        # ignore Qiskit TextProgressBar that prints to stderr
        with contextlib.redirect_stderr(io.StringIO()):
            TestLogging._run_test()
        # check that logging was handled
        self._validate_records(self._test_handler.records)

    def test_logging_to_default_handler(self):
        """logging test to file"""
        self._set_logging(True)
        # ignore Qiskit TextProgressBar that prints to stderr
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertLogs("qiskit", level="DEBUG") as qiskit_cm:
                with self.assertLogs("qiskit_nature", level="DEBUG") as nature_cm:
                    TestLogging._run_test()
        # check that logging was handled
        records = qiskit_cm.records.copy()
        records.extend(nature_cm.records)
        self._validate_records(records)

    def test_logging_to_file(self):
        """logging test to file"""
        self._set_logging(False)
        # pylint: disable=consider-using-with
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_file.close()
        os.unlink(tmp_file.name)
        file_handler = nature_logging.log_to_file(
            self._logging_dict.keys(), path=tmp_file.name, mode="w"
        )
        try:
            # ignore Qiskit TextProgressBar that prints to stderr
            with contextlib.redirect_stderr(io.StringIO()):
                TestLogging._run_test()
        finally:
            with open(tmp_file.name, encoding="utf8") as file:
                lines = file.read()
            file_handler.close()
            os.unlink(tmp_file.name)

        for name in self._logging_dict:
            self.assertTrue(f"{name}." in lines, msg=f"name {name} not found in log file.")

    @staticmethod
    def _run_test():
        """Run external test and ignore any failures. Intention is just check logging."""
        # pylint: disable=import-outside-toplevel
        from test.algorithms.excited_state_solvers.test_bosonic_esc_calculation import (
            TestBosonicESCCalculation,
        )

        unittest.TextTestRunner().run(TestBosonicESCCalculation("test_numpy_mes"))


if __name__ == "__main__":
    unittest.main()
