# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver PySCF """

import unittest
from test import QiskitNatureTestCase, requires_extra_library
from test.drivers.test_driver import TestDriver
from qiskit_nature.drivers import PySCFDriver, UnitsType


class TestDriverPySCF(QiskitNatureTestCase, TestDriver):
    """PYSCF Driver tests."""

    @requires_extra_library
    def setUp(self):
        super().setUp()
        driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.735",
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )
        self.qmolecule = driver.run()


class TestDriverPySCFMolecule(QiskitNatureTestCase, TestDriver):
    """PYSCF Driver Molecule tests."""

    @requires_extra_library
    def setUp(self):
        super().setUp()
        driver = PySCFDriver(molecule=TestDriver.MOLECULE)
        self.qmolecule = driver.run()


if __name__ == '__main__':
    unittest.main()
