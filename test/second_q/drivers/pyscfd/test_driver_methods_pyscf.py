# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver Methods PySCF """

import unittest

from test.second_q.drivers.test_driver_methods_gsc import TestDriverMethods
from qiskit.test import slow_test
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
import qiskit_nature.optionals as _optionals


class TestDriverMethodsPySCF(TestDriverMethods):
    """Driver Methods PySCF tests"""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()
        PySCFDriver(atom=self.lih)

    def test_lih_rhf(self):
        """lih rhf test"""
        driver = PySCFDriver(
            atom=self.lih,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto-3g",
            method=MethodType.RHF,
        )
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_rohf(self):
        """lih rohf test"""
        driver = PySCFDriver(
            atom=self.lih,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto-3g",
            method=MethodType.ROHF,
        )
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_uhf(self):
        """lih uhf test"""
        driver = PySCFDriver(
            atom=self.lih,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto-3g",
            method=MethodType.UHF,
        )
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_rhf_parity(self):
        """lih rhf parity test"""
        driver = PySCFDriver(
            atom=self.lih,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto-3g",
            method=MethodType.RHF,
        )
        result = self._run_driver(
            driver,
            mapper=ParityMapper(),
            transformers=[FreezeCoreTransformer()],
        )
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_rhf_parity_2q(self):
        """lih rhf parity 2q test"""
        driver = PySCFDriver(
            atom=self.lih,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto-3g",
            method=MethodType.RHF,
        )
        result = self._run_driver(
            driver,
            # NOTE: the particle number provided below already needs to take the core orbital
            # freezing into account
            mapper=ParityMapper(num_particles=(1, 1)),
            transformers=[FreezeCoreTransformer()],
        )
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_rhf_bk(self):
        """lih rhf bk test"""
        driver = PySCFDriver(
            atom=self.lih,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto-3g",
            method=MethodType.RHF,
        )
        result = self._run_driver(
            driver,
            mapper=BravyiKitaevMapper(),
            transformers=[FreezeCoreTransformer()],
        )
        self._assert_energy_and_dipole(result, "lih")

    @slow_test
    def test_oh_rohf(self):
        """oh rohf test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.ROHF,
        )
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, "oh")

    @slow_test
    def test_oh_uhf(self):
        """oh uhf test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.UHF,
        )
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, "oh")

    @slow_test
    def test_oh_rohf_parity(self):
        """oh rohf parity test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.ROHF,
        )
        result = self._run_driver(driver, mapper=ParityMapper())
        self._assert_energy_and_dipole(result, "oh")

    @slow_test
    def test_oh_rohf_parity_2q(self):
        """oh rohf parity 2q test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.ROHF,
        )
        result = self._run_driver(driver, mapper=ParityMapper(num_particles=(5, 4)))
        self._assert_energy_and_dipole(result, "oh")

    @slow_test
    def test_oh_uhf_parity(self):
        """oh uhf parity test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.UHF,
        )
        result = self._run_driver(driver, mapper=ParityMapper())
        self._assert_energy_and_dipole(result, "oh")

    @slow_test
    def test_oh_uhf_parity_2q(self):
        """oh uhf parity 2q test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.UHF,
        )
        result = self._run_driver(driver, mapper=ParityMapper(num_particles=(5, 4)))
        self._assert_energy_and_dipole(result, "oh")

    @slow_test
    def test_oh_rohf_bk(self):
        """oh rohf bk test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.ROHF,
        )
        result = self._run_driver(driver, mapper=BravyiKitaevMapper())
        self._assert_energy_and_dipole(result, "oh")

    @slow_test
    def test_oh_uhf_bk(self):
        """oh uhf bk test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=DistanceUnit.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.UHF,
        )
        result = self._run_driver(driver, mapper=BravyiKitaevMapper())
        self._assert_energy_and_dipole(result, "oh")


if __name__ == "__main__":
    unittest.main()
