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

""" Test Gaussian Forces Driver """

from __future__ import annotations

import re
import unittest

from test import QiskitNatureTestCase
from ddt import data, ddt, unpack

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.drivers import (
    GaussianForcesDriver,
    GaussianLogDriver,
)
from qiskit_nature.second_q.problems import HarmonicBasis
from qiskit_nature.exceptions import QiskitNatureError
import qiskit_nature.optionals as _optionals


@ddt
class TestDriverGaussianForces(QiskitNatureTestCase):
    """Gaussian Forces Driver tests."""

    _C01_REV_EXPECTED = [
        [352.3005875, 2, 2],
        [-352.3005875, -2, -2],
        [631.6153975, 1, 1],
        [-631.6153975, -1, -1],
        [115.653915, 4, 4],
        [-115.653915, -4, -4],
        [115.653915, 3, 3],
        [-115.653915, -3, -3],
        [-15.341901966295344, 2, 2, 2],
        [-88.2017421687633, 1, 1, 2],
        [42.675273102831454, 4, 4, 2],
        [42.675273102831454, 3, 3, 2],
        [0.420735625, 2, 2, 2, 2],
        [4.9425425, 1, 1, 2, 2],
        [1.6122932291666665, 1, 1, 1, 1],
        [-4.194299375, 4, 4, 2, 2],
        [-4.194299375, 3, 3, 2, 2],
        [-10.20589125, 4, 4, 1, 1],
        [-10.20589125, 3, 3, 1, 1],
        [2.335859166666667, 4, 4, 4, 4],
        [2.6559641666666667, 4, 4, 4, 3],
        [7.09835, 4, 4, 3, 3],
        [-2.6559641666666667, 4, 3, 3, 3],
        [2.335859166666667, 3, 3, 3, 3],
    ]

    _C01_REV_PBE_EXPECTED = [
        [353.5831025, 2, 2],
        [-353.5831025, -2, -2],
        [644.7579625, 1, 1],
        [-644.7579625, -1, -1],
        [95.344445, 4, 4],
        [-95.344445, -4, -4],
        [95.344445, 3, 3],
        [-95.344445, -3, -3],
        [-15.110404634225285, 2, 2, 2],
        [0.409219375, 2, 2, 2, 2],
        [1.5162932291666669, 1, 1, 1, 1],
        [3.5002357291666666, 4, 4, 4, 4],
        [3.5002358333333334, 3, 3, 3, 3],
        [-85.62267978593646, 1, 1, 2],
        [53.86795497291527, 4, 4, 2],
        [53.86795497291527, 3, 3, 2],
        [4.740964375, 1, 1, 2, 2],
        [-5.103499375, 4, 4, 2, 2],
        [-5.103499375, 3, 3, 2, 2],
        [-12.17633125, 4, 4, 1, 1],
        [-12.17633125, 3, 3, 1, 1],
        [3.984481666666667, 4, 4, 4, 3],
        [10.624813125, 4, 4, 3, 3],
        [-3.984481666666667, 4, 3, 3, 3],
    ]

    _A03_REV_EXPECTED = [
        [352.3005875, 2, 2],
        [-352.3005875, -2, -2],
        [631.6153975, 1, 1],
        [-631.6153975, -1, -1],
        [115.653915, 4, 4],
        [-115.653915, -4, -4],
        [115.653915, 3, 3],
        [-115.653915, -3, -3],
        [-15.341901966295344, 2, 2, 2],
        [0.4207357291666667, 2, 2, 2, 2],
        [1.6122932291666665, 1, 1, 1, 1],
        [2.2973803125, 4, 4, 4, 4],
        [2.2973803125, 3, 3, 3, 3],
        [-88.2017421687633, 1, 1, 2],
        [42.40478531359112, 4, 4, 2],
        [2.2874639206341865, 3, 3, 2],
        [4.9425425, 1, 1, 2, 2],
        [-4.194299375, 4, 4, 2, 2],
        [-4.194299375, 3, 3, 2, 2],
        [-10.20589125, 4, 4, 1, 1],
        [-10.20589125, 3, 3, 1, 1],
        [2.7821204166666664, 4, 4, 4, 3],
        [7.329224375, 4, 4, 3, 3],
        [-2.7821200000000004, 4, 3, 3, 3],
        [26.25167512727164, 4, 3, 2],
    ]

    _A03_REV_PBE_EXPECTED = [
        [353.5831025, 2, 2],
        [-353.5831025, -2, -2],
        [644.7579625, 1, 1],
        [-644.7579625, -1, -1],
        [95.344445, 4, 4],
        [-95.344445, -4, -4],
        [95.344445, 3, 3],
        [-95.344445, -3, -3],
        [-15.110404634225285, 2, 2, 2],
        [0.409219375, 2, 2, 2, 2],
        [1.5162932291666669, 1, 1, 1, 1],
        [1.6779766666666667, 4, 4, 4, 4],
        [1.6779765624999998, 3, 3, 3, 3],
        [-85.62267978593646, 1, 1, 2],
        [38.64810022630838, 4, 4, 2],
        [17.765831603142026, 3, 3, 2],
        [4.740964375, 1, 1, 2, 2],
        [-5.103499375, 4, 4, 2, 2],
        [-5.103499375, 3, 3, 2, 2],
        [-12.17633125, 4, 4, 1, 1],
        [-12.17633125, 3, 3, 1, 1],
        [3.6290025, 4, 4, 4, 3],
        [21.5583675, 4, 4, 3, 3],
        [-3.6290025, 4, 3, 3, 3],
        [69.86851040672767, 4, 3, 2],
    ]

    _B01_REV_EXPECTED = [
        [352.3005875, 2, 2],
        [-352.3005875, -2, -2],
        [631.6153975, 1, 1],
        [-631.6153975, -1, -1],
        [115.653915, 4, 4],
        [-115.653915, -4, -4],
        [115.653915, 3, 3],
        [-115.653915, -3, -3],
        [-15.341901966295344, 2, 2, 2],
        [0.4207357291666667, 2, 2, 2, 2],
        [1.6122932291666665, 1, 1, 1, 1],
        [1.8255064583333331, 4, 4, 4, 4],
        [1.8255065625, 3, 3, 3, 3],
        [-88.2017421687633, 1, 1, 2],
        [38.72849649956234, 4, 4, 2],
        [38.72849649956234, 3, 3, 2],
        [4.9425425, 1, 1, 2, 2],
        [-4.194299375, 4, 4, 2, 2],
        [-4.194299375, 3, 3, 2, 2],
        [-10.205891875, 4, 4, 1, 1],
        [-10.205891875, 3, 3, 1, 1],
        [3.507156666666667, 4, 4, 4, 3],
        [10.160466875, 4, 4, 3, 3],
        [-3.507156666666667, 4, 3, 3, 3],
    ]

    _B01_REV_PBE_EXPECTED = [
        [353.5831025, 2, 2],
        [-353.5831025, -2, -2],
        [644.7579625, 1, 1],
        [-644.7579625, -1, -1],
        [95.344445, 4, 4],
        [-95.344445, -4, -4],
        [95.344445, 3, 3],
        [-95.344445, -3, -3],
        [-15.110404634225285, 2, 2, 2],
        [0.409219375, 2, 2, 2, 2],
        [1.5162932291666669, 1, 1, 1, 1],
        [2.7346037499999998, 4, 4, 4, 4],
        [2.7346037499999998, 3, 3, 3, 3],
        [-85.62267978593646, 1, 1, 2],
        [48.886037681599355, 4, 4, 2],
        [48.886037681599355, 3, 3, 2],
        [4.740964375, 1, 1, 2, 2],
        [-5.103499375, 4, 4, 2, 2],
        [-5.103499375, 3, 3, 2, 2],
        [-12.17633125, 4, 4, 1, 1],
        [-12.17633125, 3, 3, 1, 1],
        [5.261441666666667, 4, 4, 4, 3],
        [15.218605, 4, 4, 3, 3],
        [-5.261441666666667, 4, 3, 3, 3],
    ]

    def _get_expected_values(self, pbe: bool = False):
        """Get expected values based on revision of Gaussian 16 being used."""
        jcf = "\n\n"  # Empty job control file will error out
        log_driver = GaussianLogDriver(jcf=jcf)
        version = "Not found by regex"
        try:
            _ = log_driver.run()
        except QiskitNatureError as qne:
            matched = re.search("G16Rev\\w+\\.\\w+", qne.message)
            if matched is not None:
                version = matched[0]
        if version == "G16RevA.03":
            if pbe:
                exp_vals = TestDriverGaussianForces._A03_REV_PBE_EXPECTED
            else:
                exp_vals = TestDriverGaussianForces._A03_REV_EXPECTED
        elif version == "G16RevB.01":
            if pbe:
                exp_vals = TestDriverGaussianForces._B01_REV_PBE_EXPECTED
            else:
                exp_vals = TestDriverGaussianForces._B01_REV_EXPECTED
        elif version == "G16RevC.01":
            if pbe:
                exp_vals = TestDriverGaussianForces._C01_REV_PBE_EXPECTED
            else:
                exp_vals = TestDriverGaussianForces._C01_REV_EXPECTED
        else:
            self.fail(f"Unknown gaussian version '{version}'")

        return exp_vals

    @unittest.skipIf(not _optionals.HAS_GAUSSIAN, "gaussian not available.")
    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def test_driver_jcf(self):
        """Test the driver works with job control file"""
        driver = GaussianForcesDriver(
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
        basis = HarmonicBasis([2, 2, 2, 2])
        result = driver.run(basis=basis)
        self._check_driver_result(self._get_expected_values(), result)

    @staticmethod
    def _print_raw_values(vibrational_energy):
        """Print result raw values to compare with reference"""
        print("\n[")
        for ints in vibrational_energy._vibrational_integrals.values():
            for value, indices in ints._integrals:
                line = f"[{value}"
                for ind in indices:
                    line += f", {ind}"
                line += "],"
                print(line)
        print("]\n")

    @unittest.skipIf(not _optionals.HAS_GAUSSIAN, "gaussian not available.")
    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    @data("B3LYP", "PBEPBE")
    def test_driver_molecule(self, xcf: str):
        """Test the driver works with Molecule"""
        molecule = MoleculeInfo(
            symbols=["C", "O", "O"],
            coords=[
                (-0.848629, 2.067624, 0.160992),
                (0.098816, 2.655801, -0.159738),
                (-1.796073, 1.479446, 0.481721),
            ],
            multiplicity=1,
            charge=0,
            units=DistanceUnit.ANGSTROM,
        )
        driver = GaussianForcesDriver.from_molecule(molecule, basis="6-31g", xcf=xcf)
        basis = HarmonicBasis([2, 2, 2, 2])
        result = driver.run(basis=basis)
        self._check_driver_result(self._get_expected_values(pbe=xcf == "PBEPBE"), result)

    @data(
        ("A03", _A03_REV_EXPECTED),
        ("C01", _C01_REV_EXPECTED),
    )
    @unpack
    @unittest.skipIf(not _optionals.HAS_SPARSE, "Sparse not available.")
    def test_driver_logfile(self, suffix, expected):
        """Test the driver works with logfile (Gaussian does not need to be installed)"""

        driver = GaussianForcesDriver(
            logfile=self.get_resource_path(
                f"test_driver_gaussian_log_{suffix}.txt", "second_q/drivers/gaussiand"
            )
        )

        basis = HarmonicBasis([2, 2, 2, 2])
        result = driver.run(basis=basis)
        # Log file being tested was created with revision A.03
        self._check_driver_result(expected, result)

    def _check_driver_result(self, expected_watson_data, prop):
        # TODO:
        pass


if __name__ == "__main__":
    unittest.main()
