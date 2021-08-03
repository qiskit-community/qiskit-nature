# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver Methods Gaussian """

import unittest

from test import requires_extra_library
from test.drivers.second_quantization.test_driver_methods_gsc import TestDriverMethods
from qiskit_nature.drivers.second_quantization import GaussianDriver
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer


class TestDriverMethodsGaussian(TestDriverMethods):
    """Driver Methods Gaussian tests"""

    g16_lih_config = """
# {}/sto-3g scf(conventional)

Lih molecule

0 1
Li  0.0  0.0    0.0
H   0.0  0.0    1.6

"""

    g16_oh_config = """
# {}/sto-3g scf(conventional)

Lih molecule

0 2
O   0.0  0.0    0.0
H   0.0  0.0    0.9697

"""

    @requires_extra_library
    def setUp(self):
        super().setUp()
        GaussianDriver(config=self.g16_lih_config.format("rhf"))

    def test_lih_rhf(self):
        """lih rhf test"""
        driver = GaussianDriver(config=self.g16_lih_config.format("rhf"))
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_rohf(self):
        """lih rohf test"""
        driver = GaussianDriver(config=self.g16_lih_config.format("rohf"))
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "lih")

    def test_lih_uhf(self):
        """lih uhf test"""
        driver = GaussianDriver(config=self.g16_lih_config.format("uhf"))
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "lih")

    def test_oh_rohf(self):
        """oh rohf test"""
        driver = GaussianDriver(config=self.g16_oh_config.format("rohf"))
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "oh")

    def test_oh_uhf(self):
        """oh uhf test"""
        driver = GaussianDriver(config=self.g16_oh_config.format("uhf"))
        result = self._run_driver(driver, transformers=[FreezeCoreTransformer()])
        self._assert_energy_and_dipole(result, "oh")


if __name__ == "__main__":
    unittest.main()
