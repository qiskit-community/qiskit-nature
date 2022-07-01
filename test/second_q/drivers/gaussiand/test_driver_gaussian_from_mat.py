# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
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
from test.second_q.drivers.test_driver import TestDriver
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.drivers import GaussianDriver


class TestDriverGaussianFromMat(QiskitNatureTestCase, TestDriver):
    """Gaussian Driver tests using a saved output matrix file."""

    def setUp(self):
        super().setUp()
        matfile = self.get_resource_path(
            "test_driver_gaussian_from_mat.mat", "drivers/second_q/gaussiand"
        )
        try:
            self.driver_result = GaussianDriver._parse_matrix_file(matfile)
        except QiskitNatureError:
            self.tearDown()
            self.skipTest("GAUSSIAN qcmatrixio not found")


if __name__ == "__main__":
    unittest.main()
