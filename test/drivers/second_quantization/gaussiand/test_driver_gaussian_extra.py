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
from qiskit_nature.drivers.second_quantization import GaussianDriver


# We need to have an instance so we can test function but constructor calls
# an internal method to check G16 installed. We need to replace that with
# the following dummy for things to work and we do it for each test so the
# class ends up as it was
def _check_installed():
    pass


class TestDriverGaussianExtra(QiskitNatureTestCase):
    """Gaussian Driver extra tests for driver specifics, errors etc"""

    def setUp(self):
        super().setUp()
        self.good_check = GaussianDriver.check_installed
        GaussianDriver.check_installed = _check_installed
        # We can now create a driver without the installed (check valid) test failing

    def tearDown(self):
        GaussianDriver.check_installed = self.good_check

    def test_cfg_augment(self):
        """test input configuration augmentation"""
        cfg = (
            "# rhf/sto-3g scf(conventional)\n\n"
            "h2 molecule\n\n0 1\nH   0.0  0.0    0.0\nH   0.0  0.0    0.735\n\n"
        )
        g16 = GaussianDriver(cfg)
        aug_cfg = g16._augment_config("mymatfile.mat", cfg)
        expected = (
            "# rhf/sto-3g scf(conventional)\n"
            "# Window=Full Int=NoRaff Symm=(NoInt,None)"
            " output=(matrix,i4labels,mo2el) tran=full\n\n"
            "h2 molecule\n\n0 1\nH   0.0  0.0    0.0\nH   0.0  0.0    0.735"
            "\n\nmymatfile.mat\n\n"
        )
        self.assertEqual(aug_cfg, expected)


if __name__ == "__main__":
    unittest.main()
