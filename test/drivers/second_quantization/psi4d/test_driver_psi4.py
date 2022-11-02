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

""" Test Driver PSI4 """

import unittest

from test import QiskitNatureDeprecatedTestCase
from test.drivers.second_quantization.test_driver import TestDriver
from qiskit_nature.drivers.second_quantization import (
    PSI4Driver,
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
import qiskit_nature.optionals as _optionals


class TestDriverPSI4(QiskitNatureDeprecatedTestCase, TestDriver):
    """PSI4 Driver tests."""

    @unittest.skipIf(not _optionals.HAS_PSI4, "psi4 not available.")
    def setUp(self):
        super().setUp()
        driver = PSI4Driver(
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


class TestDriverPSI4Molecule(QiskitNatureDeprecatedTestCase, TestDriver):
    """PSI4 Driver molecule tests."""

    @unittest.skipIf(not _optionals.HAS_PSI4, "psi4 not available.")
    def setUp(self):
        super().setUp()
        driver = ElectronicStructureMoleculeDriver(
            TestDriver.MOLECULE, driver_type=ElectronicStructureDriverType.PSI4
        )
        self.driver_result = driver.run()


if __name__ == "__main__":
    unittest.main()
