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

""" Test Driver Gaussian """

import unittest

from test import QiskitNatureTestCase
from test.second_q.drivers.second_quantization.test_driver import TestDriver
from qiskit_nature.second_q.drivers.second_quantization import (
    GaussianDriver,
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
import qiskit_nature.optionals as _optionals


class TestDriverGaussian(QiskitNatureTestCase, TestDriver):
    """Gaussian Driver tests."""

    @unittest.skipIf(not _optionals.HAS_GAUSSIAN, "gaussian not available.")
    def setUp(self):
        super().setUp()
        driver = GaussianDriver(
            [
                "# rhf/sto-3g scf(conventional) geom=nocrowd",
                "",
                "h2 molecule",
                "",
                "0 1",
                "H   0.0  0.0    0.0",
                "H   0.0  0.0    0.735",
                "",
            ]
        )
        self.driver_result = driver.run()


class TestDriverGaussianMolecule(QiskitNatureTestCase, TestDriver):
    """Gaussian Driver tests."""

    @unittest.skipIf(not _optionals.HAS_GAUSSIAN, "gaussian not available.")
    def setUp(self):
        super().setUp()
        driver = ElectronicStructureMoleculeDriver(
            TestDriver.MOLECULE, driver_type=ElectronicStructureDriverType.GAUSSIAN
        )
        self.driver_result = driver.run()


if __name__ == "__main__":
    unittest.main()
