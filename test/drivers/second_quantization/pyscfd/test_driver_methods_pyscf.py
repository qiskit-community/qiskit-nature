# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
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

from test.drivers.second_quantization.test_driver_methods_gsc import TestDriverMethods
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver, MethodType
from qiskit_nature.mappers.second_quantization import BravyiKitaevMapper, ParityMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
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
            unit=UnitsType.ANGSTROM,
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
            unit=UnitsType.ANGSTROM,
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
            unit=UnitsType.ANGSTROM,
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
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto-3g",
            method=MethodType.RHF,
        )
        result = self._run_driver(
            driver,
            converter=QubitConverter(ParityMapper()),
            transformers=[FreezeCoreTransformer()],
        )
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_rhf_parity_2q(self):
        """lih rhf parity 2q test"""
        driver = PySCFDriver(
            atom=self.lih,
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto-3g",
            method=MethodType.RHF,
        )
        result = self._run_driver(
            driver,
            converter=QubitConverter(ParityMapper(), two_qubit_reduction=True),
            transformers=[FreezeCoreTransformer()],
        )
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_rhf_bk(self):
        """lih rhf bk test"""
        driver = PySCFDriver(
            atom=self.lih,
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto-3g",
            method=MethodType.RHF,
        )
        result = self._run_driver(
            driver,
            converter=QubitConverter(BravyiKitaevMapper()),
            transformers=[FreezeCoreTransformer()],
        )
        self._assert_energy_and_dipole(result, "lih")

    def test_oh_rohf(self):
        """oh rohf test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.ROHF,
        )
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, "oh")

    def test_oh_uhf(self):
        """oh uhf test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.UHF,
        )
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, "oh")

    def test_oh_rohf_parity(self):
        """oh rohf parity test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.ROHF,
        )
        result = self._run_driver(driver, converter=QubitConverter(ParityMapper()))
        self._assert_energy_and_dipole(result, "oh")

    def test_oh_rohf_parity_2q(self):
        """oh rohf parity 2q test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.ROHF,
        )
        result = self._run_driver(
            driver, converter=QubitConverter(ParityMapper(), two_qubit_reduction=True)
        )
        self._assert_energy_and_dipole(result, "oh")

    def test_oh_uhf_parity(self):
        """oh uhf parity test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.UHF,
        )
        result = self._run_driver(driver, converter=QubitConverter(ParityMapper()))
        self._assert_energy_and_dipole(result, "oh")

    def test_oh_uhf_parity_2q(self):
        """oh uhf parity 2q test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.UHF,
        )
        result = self._run_driver(
            driver, converter=QubitConverter(ParityMapper(), two_qubit_reduction=True)
        )
        self._assert_energy_and_dipole(result, "oh")

    def test_oh_rohf_bk(self):
        """oh rohf bk test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.ROHF,
        )
        result = self._run_driver(driver, converter=QubitConverter(BravyiKitaevMapper()))
        self._assert_energy_and_dipole(result, "oh")

    def test_oh_uhf_bk(self):
        """oh uhf bk test"""
        driver = PySCFDriver(
            atom=self.o_h,
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=1,
            basis="sto-3g",
            method=MethodType.UHF,
        )
        result = self._run_driver(driver, converter=QubitConverter(BravyiKitaevMapper()))
        self._assert_energy_and_dipole(result, "oh")


if __name__ == "__main__":
    unittest.main()
