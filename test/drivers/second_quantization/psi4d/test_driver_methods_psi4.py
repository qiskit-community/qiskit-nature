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

""" Test Driver Methods PSI4 """

import unittest

from test.drivers.second_quantization.test_driver_methods_gsc import TestDriverMethods
from qiskit_nature.drivers.second_quantization import PSI4Driver
from qiskit_nature.second_q.problems.electronic import FreezeCoreTransformer
import qiskit_nature.optionals as _optionals


class TestDriverMethodsPSI4(TestDriverMethods):
    """Driver Methods PSI4 tests"""

    psi4_lih_config = """
molecule mol {{
   0 1
   Li  0.0 0.0 0.0
   H   0.0 0.0 1.6
}}

set {{
      basis sto-3g
      scf_type pk
      reference {}
}}
"""

    psi4_oh_config = """
molecule mol {{
   0 2
   O  0.0 0.0 0.0
   H  0.0 0.0 0.9697
}}

set {{
      basis sto-3g
      scf_type pk
      reference {}
}}
"""

    @unittest.skipIf(not _optionals.HAS_PSI4, "psi4 not available.")
    def setUp(self):
        super().setUp()
        PSI4Driver(config=self.psi4_lih_config.format("rhf"))

    def test_lih_rhf(self):
        """lih rhf test"""
        driver = PSI4Driver(config=self.psi4_lih_config.format("rhf"))
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_rohf(self):
        """lih rohf test"""
        driver = PSI4Driver(config=self.psi4_lih_config.format("rohf"))
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_uhf(self):
        """lih uhf test"""
        driver = PSI4Driver(config=self.psi4_lih_config.format("uhf"))
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "lih")

    def test_oh_rohf(self):
        """oh rohf test"""
        driver = PSI4Driver(config=self.psi4_oh_config.format("rohf"))
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, "oh")

    def test_oh_uhf(self):
        """oh uhf test"""
        driver = PSI4Driver(config=self.psi4_oh_config.format("uhf"))
        result = self._run_driver(driver)
        self._assert_energy_and_dipole(result, "oh")


if __name__ == "__main__":
    unittest.main()
