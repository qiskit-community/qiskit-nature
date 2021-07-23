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

""" Test Driver PSI4 """

import unittest

from test import QiskitNatureTestCase, requires_extra_library
from test.drivers.second_quantization.test_driver import TestDriver
from qiskit_nature.drivers.second_quantization import (
    PSI4Driver,
    FermionicDriverType,
    FermionicMoleculeDriver,
)


class TestDriverPSI4(QiskitNatureTestCase, TestDriver):
    """PSI4 Driver tests."""

    @requires_extra_library
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
        self.qmolecule = driver.run()


class TestDriverPSI4Molecule(QiskitNatureTestCase, TestDriver):
    """PSI4 Driver molecule tests."""

    @requires_extra_library
    def setUp(self):
        super().setUp()
        driver = FermionicMoleculeDriver(TestDriver.MOLECULE, driver_type=FermionicDriverType.PSI4)
        self.qmolecule = driver.run()


if __name__ == "__main__":
    unittest.main()
