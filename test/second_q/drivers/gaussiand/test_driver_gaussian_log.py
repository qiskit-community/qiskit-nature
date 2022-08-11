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

""" Test Gaussian Log Driver """

import unittest

from test import QiskitNatureTestCase
import numpy as np

from qiskit_nature.hdf5 import load_from_hdf5
from qiskit_nature.second_q.drivers import GaussianLogDriver, GaussianLogResult
from qiskit_nature.second_q.properties import VibrationalEnergy
from qiskit_nature.second_q.properties.integrals import (
    VibrationalIntegrals,
)
import qiskit_nature.optionals as _optionals


class TestDriverGaussianLog(QiskitNatureTestCase):
    """Gaussian Log Driver tests."""

    def setUp(self):
        super().setUp()
        self.logfile = self.get_resource_path(
            "test_driver_gaussian_log_A03.txt", "second_q/drivers/gaussiand"
        )

    @unittest.skipIf(not _optionals.HAS_GAUSSIAN, "gaussian not available.")
    def test_log_driver(self):
        """Test the driver itself creates log and we can get a result"""
        driver = GaussianLogDriver(
            [
                "#p B3LYP/6-31g Freq=(Anharm) Int=Ultrafine SCF=VeryTight",
                "",
                "CO2 geometry optimization B3LYP/6-31g",
                "",
                "0 1",
                "C  -0.848629  2.067624  0.160992",
                "O   0.098816  2.655801 -0.159738",
                "O  -1.796073  1.479446  0.481721",
                "",
                "",
            ]
        )
        result = driver.run()
        qfc = result.quadratic_force_constants
        expected = [
            ("1", "1", 1409.20235, 1.17003, 0.07515),
            ("2", "2", 2526.46159, 3.76076, 0.24156),
            ("3a", "3a", 462.61566, 0.12609, 0.0081),
            ("3b", "3b", 462.61566, 0.12609, 0.0081),
        ]
        self.assertListEqual(qfc, expected)

    # These tests check the gaussian log result and the parsing from a partial log file that is
    # located with the tests so that this aspect of the code can be tested independent of
    # Gaussian 16 being installed.

    def test_gaussian_log_result_file(self):
        """Test result from file"""
        result = GaussianLogResult(self.logfile)
        with open(self.logfile, "r", encoding="utf8") as file:
            lines = file.read().split("\n")

        with self.subTest("Check list of lines"):
            self.assertListEqual(result.log, lines)

        with self.subTest("Check as string"):
            line = "\n".join(lines)
            self.assertEqual(str(result), line)

    def test_gaussian_log_result_list(self):
        """Test result from list of strings"""
        with open(self.logfile, "r", encoding="utf8") as file:
            lines = file.read().split("\n")
        result = GaussianLogResult(lines)
        self.assertListEqual(result.log, lines)

    def test_gaussian_log_result_string(self):
        """Test result from string"""
        with open(self.logfile, "r", encoding="utf8") as file:
            line = file.read()
        result = GaussianLogResult(line)
        self.assertListEqual(result.log, line.split("\n"))

    def test_multi_line_data(self):
        """Test if data is found on multiple lines.

        This is a regression test against https://github.com/Qiskit/qiskit-nature/issues/737
        """
        polyyne_log = self.get_resource_path(
            "test_driver_gaussian_log_polyyne_2.txt", "second_q/drivers/gaussiand"
        )
        result = GaussianLogResult(polyyne_log)
        vib_energy = result.get_vibrational_energy()
        self.assertTrue(isinstance(vib_energy, VibrationalEnergy))

    def test_quadratic_force_constants(self):
        """Test quadratic force constants"""
        result = GaussianLogResult(self.logfile)
        qfc = result.quadratic_force_constants
        expected = [
            ("1", "1", 1409.20235, 1.17003, 0.07515),
            ("2", "2", 2526.46159, 3.76076, 0.24156),
            ("3a", "3a", 462.61566, 0.12609, 0.0081),
            ("3b", "3b", 462.61566, 0.12609, 0.0081),
        ]
        self.assertListEqual(qfc, expected)

    def test_cubic_force_constants(self):
        """Test cubic force constants"""
        result = GaussianLogResult(self.logfile)
        cfc = result.cubic_force_constants
        expected = [
            ("1", "1", "1", -260.36071, -1.39757, -0.0475),
            ("2", "2", "1", -498.9444, -4.80163, -0.1632),
            ("3a", "3a", "1", 239.87769, 0.4227, 0.01437),
            ("3a", "3b", "1", 74.25095, 0.13084, 0.00445),
            ("3b", "3b", "1", 12.93985, 0.0228, 0.00078),
        ]
        self.assertListEqual(cfc, expected)

    def test_quartic_force_constants(self):
        """Test quartic force constants"""
        result = GaussianLogResult(self.logfile)
        qfc = result.quartic_force_constants
        expected = [
            ("1", "1", "1", "1", 40.39063, 1.40169, 0.02521),
            ("2", "2", "1", "1", 79.08068, 4.92017, 0.0885),
            ("2", "2", "2", "2", 154.78015, 17.26491, 0.31053),
            ("3a", "3a", "1", "1", -67.10879, -0.76453, -0.01375),
            ("3b", "3b", "1", "1", -67.10879, -0.76453, -0.01375),
            ("3a", "3a", "2", "2", -163.29426, -3.33524, -0.05999),
            ("3b", "3b", "2", "2", -163.29426, -3.33524, -0.05999),
            ("3a", "3a", "3a", "3a", 220.54851, 0.82484, 0.01484),
            ("3a", "3a", "3a", "3b", 66.77089, 0.24972, 0.00449),
            ("3a", "3a", "3b", "3b", 117.26759, 0.43857, 0.00789),
            ("3a", "3b", "3b", "3b", -66.77088, -0.24972, -0.00449),
            ("3b", "3b", "3b", "3b", 220.54851, 0.82484, 0.01484),
        ]
        self.assertListEqual(qfc, expected)

    def test_vibrational_energy(self):
        """Test the VibrationalEnergy."""
        result = GaussianLogResult(self.logfile)
        vib_energy = result.get_vibrational_energy()
        expected = load_from_hdf5(
            self.get_resource_path(
                "test_driver_gaussian_log_vibrational_energy.hdf5",
                "second_q/drivers/gaussiand",
            )
        )

        self.addTypeEqualityFunc(VibrationalIntegrals, self.compare_vibrational_integral)
        self.addTypeEqualityFunc(VibrationalEnergy, self.compare_vibrational_energy)
        self.assertEqual(vib_energy, expected)

    def compare_vibrational_integral(
        self, first: VibrationalIntegrals, second: VibrationalIntegrals, msg: str = None
    ) -> None:
        """Compares two VibrationalIntegral instances."""
        if first.name != second.name:
            raise self.failureException(msg)

        if first._num_body_terms != second._num_body_terms:
            raise self.failureException(msg)

        for f_int, s_int in zip(first._integrals, second._integrals):
            if not np.isclose(f_int[0], s_int[0]):
                raise self.failureException(msg)

            if not all(f == s for f, s in zip(f_int[1:], s_int[1:])):
                raise self.failureException(msg)

    def compare_vibrational_energy(
        self, first: VibrationalEnergy, second: VibrationalEnergy, msg: str = None
    ) -> None:
        # pylint: disable=unused-argument
        """Compares two VibrationalEnergy instances."""
        for f_ints, s_ints in zip(
            first._vibrational_integrals.values(), second._vibrational_integrals.values()
        ):
            self.compare_vibrational_integral(f_ints, s_ints)


if __name__ == "__main__":
    unittest.main()
