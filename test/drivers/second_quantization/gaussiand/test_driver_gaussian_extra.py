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
from qiskit_nature.drivers.second_quantization import GaussianDriver


class TestDriverGaussianExtra(QiskitNatureTestCase):
    """Gaussian Driver extra tests for driver specifics, errors etc"""

    def test_cfg_augment(self):
        """test input configuration augmentation"""
        cfg = (
            "# rhf/sto-3g scf(conventional)\n\n"
            "h2 molecule\n\n0 1\nH   0.0  0.0    0.0\nH   0.0  0.0    0.735\n\n"
        )
        aug_cfg = GaussianDriver._augment_config("mymatfile.mat", cfg)
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
