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

from typing import cast

import unittest
from test import QiskitNatureTestCase
from test.drivers.second_quantization.test_driver import TestDriver
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import (
    PySCFDriver,
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operator_factories.electronic import ElectronicEnergy
import qiskit_nature.optionals as _optionals


class TestDriverPySCF(QiskitNatureTestCase, TestDriver):
    """PYSCF Driver tests."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.735",
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )
        self.driver_result = driver.run()

    def test_h3(self):
        """Test for H3 chain, see also https://github.com/Qiskit/qiskit-aqua/issues/1148."""
        atom = "H 0 0 0; H 0 0 1; H 0 0 2"
        driver = PySCFDriver(atom=atom, unit=UnitsType.ANGSTROM, charge=0, spin=1, basis="sto3g")
        driver_result = driver.run()
        electronic_energy = cast(ElectronicEnergy, driver_result.get_property(ElectronicEnergy))
        self.assertAlmostEqual(electronic_energy.reference_energy, -1.523996200246108, places=5)

    def test_h4(self):
        """Test for H4 chain"""
        atom = "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3"
        driver = PySCFDriver(atom=atom, unit=UnitsType.ANGSTROM, charge=0, spin=0, basis="sto3g")
        driver_result = driver.run()
        electronic_energy = cast(ElectronicEnergy, driver_result.get_property(ElectronicEnergy))
        self.assertAlmostEqual(electronic_energy.reference_energy, -2.09854593699776, places=5)

    def test_invalid_atom_type(self):
        """Atom is string with ; separator or list of string"""
        with self.assertRaises(QiskitNatureError):
            PySCFDriver(atom=("H", 0, 0, 0))

    def test_list_atom(self):
        """Check input with list of strings"""
        atom = ["H 0 0 0", "H 0 0 1"]
        driver = PySCFDriver(atom=atom, unit=UnitsType.ANGSTROM, charge=0, spin=0, basis="sto3g")
        driver_result = driver.run()
        electronic_energy = cast(ElectronicEnergy, driver_result.get_property(ElectronicEnergy))
        self.assertAlmostEqual(electronic_energy.reference_energy, -1.0661086493179366, places=5)

    def test_zmatrix(self):
        """Check z-matrix input"""
        atom = "H; H 1 1.0"
        driver = PySCFDriver(atom=atom, unit=UnitsType.ANGSTROM, charge=0, spin=0, basis="sto3g")
        driver_result = driver.run()
        electronic_energy = cast(ElectronicEnergy, driver_result.get_property(ElectronicEnergy))
        self.assertAlmostEqual(electronic_energy.reference_energy, -1.0661086493179366, places=5)


class TestDriverPySCFMolecule(QiskitNatureTestCase, TestDriver):
    """PYSCF Driver Molecule tests."""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        driver = ElectronicStructureMoleculeDriver(
            TestDriver.MOLECULE, driver_type=ElectronicStructureDriverType.PYSCF
        )
        self.driver_result = driver.run()


if __name__ == "__main__":
    unittest.main()
