# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
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
from qiskit_nature.drivers import PySCFDriver, UnitsType
from qiskit_nature import QiskitNatureError


class TestDriverPySCFExtra(QiskitNatureTestCase):
    """PySCF Driver extra tests for driver specifics, errors etc"""

    @requires_extra_library
    def setUp(self):
        super().setUp()
        PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.735",
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )

    def test_h3(self):
        """Test for H3 chain, see also issue 1148"""
        atom = "H 0 0 0; H 0 0 1; H 0 0 2"
        driver = PySCFDriver(atom=atom, unit=UnitsType.ANGSTROM, charge=0, spin=1, basis="sto3g")
        molecule = driver.run()
        self.assertAlmostEqual(molecule.hf_energy, -1.523996200246108, places=5)

    def test_h4(self):
        """Test for H4 chain"""
        atom = "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3"
        driver = PySCFDriver(atom=atom, unit=UnitsType.ANGSTROM, charge=0, spin=0, basis="sto3g")
        molecule = driver.run()
        self.assertAlmostEqual(molecule.hf_energy, -2.09854593699776, places=5)

    def test_invalid_atom_type(self):
        """Atom is string with ; separator or list of string"""
        with self.assertRaises(QiskitNatureError):
            PySCFDriver(atom=("H", 0, 0, 0))

    def test_list_atom(self):
        """Check input with list of strings"""
        atom = ["H 0 0 0", "H 0 0 1"]
        driver = PySCFDriver(atom=atom, unit=UnitsType.ANGSTROM, charge=0, spin=0, basis="sto3g")
        molecule = driver.run()
        self.assertAlmostEqual(molecule.hf_energy, -1.0661086493179366, places=5)

    def test_zmatrix(self):
        """Check z-matrix input"""
        atom = "H; H 1 1.0"
        driver = PySCFDriver(atom=atom, unit=UnitsType.ANGSTROM, charge=0, spin=0, basis="sto3g")
        molecule = driver.run()
        self.assertAlmostEqual(molecule.hf_energy, -1.0661086493179366, places=5)


if __name__ == "__main__":
    unittest.main()
