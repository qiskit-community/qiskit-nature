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

""" Test Driver Gaussian internals - does not require Gaussian installed """

import unittest

from test import QiskitNatureTestCase
from test.drivers.second_quantization.test_driver import TestDriver
from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers.second_quantization import GaussianDriver


# We need to have an instance so we can test function but constructor calls
# an internal method to check G16 installed. We need to replace that with
# the following dummy for things to work and we do it for each test so the
# class ends up as it was
def _check_valid():
    pass


class TestDriverGaussianFromMat(QiskitNatureTestCase, TestDriver):
    """Gaussian Driver tests using a saved output matrix file."""

    def setUp(self):
        super().setUp()
        self.good_check = GaussianDriver._check_valid
        GaussianDriver._check_valid = _check_valid
        # We can now create a driver without the installed (check valid) test failing
        # and create a qmolecule from the saved output matrix file. This will test the
        # parsing of it into the qmolecule is correct.
        g16 = GaussianDriver()
        matfile = self.get_resource_path("test_driver_gaussian_from_mat.mat", "drivers/gaussiand")
        try:
            self.qmolecule = g16._parse_matrix_file(matfile)
        except QiskitNatureError:
            self.tearDown()
            self.skipTest("GAUSSIAN qcmatrixio not found")

    def tearDown(self):
        GaussianDriver._check_valid = self.good_check


if __name__ == "__main__":
    unittest.main()
