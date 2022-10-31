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

""" Test Driver Psi4 """

import unittest

from test import QiskitNatureTestCase
from test.second_q.drivers.test_driver import TestDriver
from qiskit_nature.second_q.drivers import Psi4Driver
import qiskit_nature.optionals as _optionals


class TestDriverPsi4(QiskitNatureTestCase, TestDriver):
    """Psi4 Driver tests."""

    @unittest.skipIf(not _optionals.HAS_PSI4, "psi4 not available.")
    def setUp(self):
        super().setUp()
        driver = Psi4Driver(
            [
                "molecule h2 {",
                "  0 1",
                "  H  0.0 0.0 0.0",
                "  H  0.0 0.0 0.735",
                "  no_com",
                "  no_reorient",
                "}",
                "",
                "set {",
                "  basis sto-3g",
                "  scf_type pk",
                "}",
            ]
        )
        self.driver_result = driver.run()


class TestDriverPSI4Molecule(QiskitNatureTestCase, TestDriver):
    """PSI4 Driver molecule tests."""

    @unittest.skipIf(not _optionals.HAS_PSI4, "psi4 not available.")
    def setUp(self):
        super().setUp()
        driver = Psi4Driver.from_molecule(TestDriver.MOLECULE)
        self.driver_result = driver.run()


if __name__ == "__main__":
    unittest.main()
