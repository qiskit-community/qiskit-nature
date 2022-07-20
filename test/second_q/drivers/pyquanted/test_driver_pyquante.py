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

""" Test Driver PyQuante """

import unittest
from test import QiskitNatureTestCase
from test.second_q.drivers.test_driver import TestDriver
from qiskit_nature.second_q.drivers import UnitsType
from qiskit_nature.second_q.drivers import (
    PyQuanteDriver,
    BasisType,
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
import qiskit_nature.optionals as _optionals


class TestDriverPyQuante(QiskitNatureTestCase, TestDriver):
    """PYQUANTE Driver tests."""

    @unittest.skipIf(not _optionals.HAS_PYQUANTE2, "pyquante2 not available.")
    def setUp(self):
        super().setUp()
        driver = PyQuanteDriver(
            atoms="H .0 .0 .0; H .0 .0 0.735",
            units=UnitsType.ANGSTROM,
            charge=0,
            multiplicity=1,
            basis=BasisType.BSTO3G,
        )
        self.driver_result = driver.run()


class TestDriverPyQuanteMolecule(QiskitNatureTestCase, TestDriver):
    """PYQUANTE Driver molecule tests."""

    @unittest.skipIf(not _optionals.HAS_PYQUANTE2, "pyquante2 not available.")
    def setUp(self):
        super().setUp()
        driver = ElectronicStructureMoleculeDriver(
            TestDriver.MOLECULE, driver_type=ElectronicStructureDriverType.PYQUANTE
        )
        self.driver_result = driver.run()


if __name__ == "__main__":
    unittest.main()
