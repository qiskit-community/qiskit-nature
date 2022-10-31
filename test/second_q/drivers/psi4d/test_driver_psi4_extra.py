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

""" Test Driver Psi4 Extra """

import unittest

from test import QiskitNatureTestCase
import numpy as np
from qiskit_nature.second_q.drivers import Psi4Driver
from qiskit_nature import QiskitNatureError
import qiskit_nature.optionals as _optionals


class TestDriverPsi4Extra(QiskitNatureTestCase):
    """Psi4 Driver extra tests for driver specifics, errors etc"""

    @unittest.skipIf(not _optionals.HAS_PSI4, "psi4 not available.")
    def setUp(self):
        super().setUp()

    def test_input_format_list(self):
        """input as a list"""
        driver = Psi4Driver(
            [
                "molecule h2 {",
                "  0 1",
                "  H  0.0 0.0 0.0",
                "  H  0.0 0.0 0.735",
                "  no_com",
                "  no_reorient",
                "}",
                "",
                "set {",
                "  basis sto-3g",
                "  scf_type pk",
                "}",
            ]
        )
        driver_result = driver.run()
        with self.subTest("energy"):
            self.assertAlmostEqual(driver_result.reference_energy, -1.117, places=3)
        with self.subTest("masses"):
            np.testing.assert_array_almost_equal(
                driver_result.molecule.masses, [1.00782503, 1.00782503]
            )

    def test_input_format_string(self):
        """input as a multi line string"""
        cfg = """
molecule h2 {
0 1
H  0.0 0.0 0.0
H  0.0 0.0 0.735
no_com
no_reorient
}
set {
basis sto-3g
scf_type pk
}
"""
        driver = Psi4Driver(cfg)
        driver_result = driver.run()
        with self.subTest("energy"):
            self.assertAlmostEqual(driver_result.reference_energy, -1.117, places=3)
        with self.subTest("masses"):
            np.testing.assert_array_almost_equal(
                driver_result.molecule.masses, [1.00782503, 1.00782503]
            )

    def test_input_format_fail(self):
        """input type failure"""
        with self.assertRaises(QiskitNatureError):
            _ = Psi4Driver(1.000)

    def test_psi4_failure(self):
        """check we catch psi4 failures (bad scf type used here)"""
        bad_cfg = """
molecule h2 {
0 1
H  0.0 0.0 0.0
H  0.0 0.0 0.735
no_com
no_reorient
}
set {
basis sto-3g
scf_type unknown
}
"""
        driver = Psi4Driver(bad_cfg)
        with self.assertRaises(QiskitNatureError) as ctxmgr:
            _ = driver.run()
        self.assertTrue(str(ctxmgr.exception).startswith("'psi4 process return code"))


if __name__ == "__main__":
    unittest.main()
