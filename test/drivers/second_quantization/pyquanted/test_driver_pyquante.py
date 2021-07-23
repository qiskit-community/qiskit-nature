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

""" Test Driver PyQuante """

import unittest
from test import QiskitNatureTestCase, requires_extra_library
from test.drivers.second_quantization.test_driver import TestDriver
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import (
    PyQuanteDriver,
    BasisType,
    FermionicDriverType,
    FermionicMoleculeDriver,
)


class TestDriverPyQuante(QiskitNatureTestCase, TestDriver):
    """PYQUANTE Driver tests."""

    @requires_extra_library
    def setUp(self):
        super().setUp()
        driver = PyQuanteDriver(
            atoms="H .0 .0 .0; H .0 .0 0.735",
            units=UnitsType.ANGSTROM,
            charge=0,
            multiplicity=1,
            basis=BasisType.BSTO3G,
        )
        self.qmolecule = driver.run()


class TestDriverPyQuanteMolecule(QiskitNatureTestCase, TestDriver):
    """PYQUANTE Driver molecule tests."""

    @requires_extra_library
    def setUp(self):
        super().setUp()
        driver = FermionicMoleculeDriver(
            TestDriver.MOLECULE, driver_type=FermionicDriverType.PYQUANTE
        )
        self.qmolecule = driver.run()


if __name__ == "__main__":
    unittest.main()
