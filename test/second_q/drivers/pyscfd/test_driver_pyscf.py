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

""" Test Driver PySCF """

import unittest
from test import QiskitNatureTestCase
from test.second_q.drivers.test_driver import TestDriver

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature import QiskitNatureError
import qiskit_nature.optionals as _optionals


class TestDriverPySCF(QiskitNatureTestCase, TestDriver):
    """PYSCF Driver tests."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.735",
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )
        self.driver_result = driver.run()

    def test_h3(self):
        """Test for H3 chain, see also https://github.com/Qiskit/qiskit-aqua/issues/1148."""
        atom = "H 0 0 0; H 0 0 1; H 0 0 2"
        driver = PySCFDriver(atom=atom, unit=DistanceUnit.ANGSTROM, charge=0, spin=1, basis="sto3g")
        driver_result = driver.run()
        self.assertAlmostEqual(driver_result.reference_energy, -1.523996200246108, places=5)

    def test_h4(self):
        """Test for H4 chain"""
        atom = "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3"
        driver = PySCFDriver(atom=atom, unit=DistanceUnit.ANGSTROM, charge=0, spin=0, basis="sto3g")
        driver_result = driver.run()
        self.assertAlmostEqual(driver_result.reference_energy, -2.09854593699776, places=5)

    def test_invalid_atom_type(self):
        """Atom is string with ; separator or list of string"""
        with self.assertRaises(QiskitNatureError):
            PySCFDriver(atom=("H", 0, 0, 0))

    def test_list_atom(self):
        """Check input with list of strings"""
        atom = ["H 0 0 0", "H 0 0 1"]
        driver = PySCFDriver(atom=atom, unit=DistanceUnit.ANGSTROM, charge=0, spin=0, basis="sto3g")
        driver_result = driver.run()
        self.assertAlmostEqual(driver_result.reference_energy, -1.0661086493179366, places=5)

    def test_zmatrix(self):
        """Check z-matrix input"""
        atom = "H; H 1 1.0"
        driver = PySCFDriver(atom=atom, unit=DistanceUnit.ANGSTROM, charge=0, spin=0, basis="sto3g")
        driver_result = driver.run()
        self.assertAlmostEqual(driver_result.reference_energy, -1.0661086493179366, places=5)


if __name__ == "__main__":
    unittest.main()
