# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
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

from ddt import ddt, data

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature import QiskitNatureError
from qiskit_nature.settings import settings
import qiskit_nature.optionals as _optionals


@ddt
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

    @data(True, False)
    def test_h3(self, use_symmetry_reduced_integrals: bool):
        """Test for H3 chain, see also https://github.com/Qiskit/qiskit-aqua/issues/1148."""
        prev_settings = settings.use_symmetry_reduced_integrals
        settings.use_symmetry_reduced_integrals = use_symmetry_reduced_integrals
        try:
            atom = "H 0 0 0; H 0 0 1; H 0 0 2"
            driver = PySCFDriver(
                atom=atom, unit=DistanceUnit.ANGSTROM, charge=0, spin=1, basis="sto3g"
            )
            driver_result = driver.run()
            self.assertAlmostEqual(driver_result.reference_energy, -1.523996200246108, places=5)
        finally:
            settings.use_symmetry_reduced_integrals = prev_settings

    @data(True, False)
    def test_h4(self, use_symmetry_reduced_integrals: bool):
        """Test for H4 chain"""
        prev_settings = settings.use_symmetry_reduced_integrals
        settings.use_symmetry_reduced_integrals = use_symmetry_reduced_integrals
        try:
            atom = "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3"
            driver = PySCFDriver(
                atom=atom, unit=DistanceUnit.ANGSTROM, charge=0, spin=0, basis="sto3g"
            )
            driver_result = driver.run()
            self.assertAlmostEqual(driver_result.reference_energy, -2.09854593699776, places=5)
        finally:
            settings.use_symmetry_reduced_integrals = prev_settings

    def test_invalid_atom_type(self):
        """Atom is string with ; separator or list of string"""
        with self.assertRaises(QiskitNatureError):
            PySCFDriver(atom=("H", 0, 0, 0))

    @data(True, False)
    def test_list_atom(self, use_symmetry_reduced_integrals: bool):
        """Check input with list of strings"""
        prev_settings = settings.use_symmetry_reduced_integrals
        settings.use_symmetry_reduced_integrals = use_symmetry_reduced_integrals
        try:
            atom = ["H 0 0 0", "H 0 0 1"]
            driver = PySCFDriver(
                atom=atom, unit=DistanceUnit.ANGSTROM, charge=0, spin=0, basis="sto3g"
            )
            driver_result = driver.run()
            self.assertAlmostEqual(driver_result.reference_energy, -1.0661086493179366, places=5)
        finally:
            settings.use_symmetry_reduced_integrals = prev_settings

    @data(True, False)
    def test_zmatrix(self, use_symmetry_reduced_integrals: bool):
        """Check z-matrix input"""
        prev_settings = settings.use_symmetry_reduced_integrals
        settings.use_symmetry_reduced_integrals = use_symmetry_reduced_integrals
        try:
            atom = "H; H 1 1.0"
            driver = PySCFDriver(
                atom=atom, unit=DistanceUnit.ANGSTROM, charge=0, spin=0, basis="sto3g"
            )
            driver_result = driver.run()
            self.assertAlmostEqual(driver_result.reference_energy, -1.0661086493179366, places=5)
        finally:
            settings.use_symmetry_reduced_integrals = prev_settings


if __name__ == "__main__":
    unittest.main()
