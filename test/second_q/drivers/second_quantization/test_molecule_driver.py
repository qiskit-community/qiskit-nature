# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Molecule Driver """

from typing import cast

import re
import unittest
import warnings

from test import QiskitNatureTestCase
from ddt import ddt, data

from qiskit.exceptions import MissingOptionalLibraryError

from qiskit_nature import QiskitNatureError
from qiskit_nature.drivers.second_quantization import (
    MethodType,
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
    VibrationalStructureDriverType,
    VibrationalStructureMoleculeDriver,
    GaussianLogDriver,
)
from qiskit_nature.drivers import Molecule, WatsonHamiltonian
from qiskit_nature.exceptions import UnsupportMethodError
from qiskit_nature.second_q.operator_factories.electronic import ElectronicEnergy
from qiskit_nature.second_q.operator_factories.vibrational import VibrationalEnergy
import qiskit_nature.optionals as _optionals


@ddt
class TestElectronicStructureMoleculeDriver(QiskitNatureTestCase):
    """Electronic structure Molecule Driver tests."""

    def setUp(self):
        super().setUp()
        self._molecule = Molecule(
            geometry=[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.735])],
            multiplicity=1,
            charge=0,
        )

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def test_invalid_kwarg(self):
        """test invalid kwarg"""
        driver = ElectronicStructureMoleculeDriver(
            self._molecule,
            basis="sto3g",
            driver_type=ElectronicStructureDriverType.PYSCF,
            driver_kwargs={"max_cycle": 0},
        )
        with self.assertRaises(ValueError):
            _ = driver.run()

    @data(
        (ElectronicStructureDriverType.AUTO, -1.1169989967540044),
        (ElectronicStructureDriverType.PYSCF, -1.1169989967540044),
        (ElectronicStructureDriverType.PSI4, -1.1169989967389082),
        (ElectronicStructureDriverType.PYQUANTE, -1.1169989925292956),
        (ElectronicStructureDriverType.GAUSSIAN, -1.1169989967389082),
    )
    def test_driver(self, config):
        """test driver"""
        driver_type, hf_energy = config
        driver = ElectronicStructureMoleculeDriver(
            self._molecule, basis="sto3g", driver_type=driver_type
        )
        try:
            driver_result = driver.run()
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
        electronic_energy = cast(ElectronicEnergy, driver_result.get_property(ElectronicEnergy))
        self.assertAlmostEqual(electronic_energy.reference_energy, hf_energy, places=5)

    @data(
        ElectronicStructureDriverType.PSI4,
        ElectronicStructureDriverType.PYQUANTE,
        ElectronicStructureDriverType.GAUSSIAN,
    )
    def test_unsupported_method(self, driver_type):
        """test unsupported methods"""
        for method in [MethodType.RKS, MethodType.ROKS, MethodType.UKS]:
            driver = ElectronicStructureMoleculeDriver(
                self._molecule, basis="sto3g", method=method, driver_type=driver_type
            )
            with self.assertRaises(UnsupportMethodError):
                try:
                    _ = driver.run()
                except MissingOptionalLibraryError as ex:
                    self.skipTest(str(ex))


@ddt
class TestVibrationalStructureMoleculeDriver(QiskitNatureTestCase):
    """Vibrational structure Molecule Driver tests."""

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
        [-88.2017421687633, 1, 1, 2],
        [42.40478531359112, 4, 4, 2],
        [26.25167512727164, 4, 3, 2],
        [2.2874639206341865, 3, 3, 2],
        [0.4207357291666667, 2, 2, 2, 2],
        [4.9425425, 1, 1, 2, 2],
        [1.6122932291666665, 1, 1, 1, 1],
        [-4.194299375, 4, 4, 2, 2],
        [-4.194299375, 3, 3, 2, 2],
        [-10.20589125, 4, 4, 1, 1],
        [-10.20589125, 3, 3, 1, 1],
        [2.2973803125, 4, 4, 4, 4],
        [2.7821204166666664, 4, 4, 4, 3],
        [7.329224375, 4, 4, 3, 3],
        [-2.7821200000000004, 4, 3, 3, 3],
        [2.2973803125, 3, 3, 3, 3],
    ]

    def _get_expected_values(self):
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
            exp_vals = TestVibrationalStructureMoleculeDriver._A03_REV_EXPECTED
        elif version == "G16RevB.01":
            exp_vals = TestVibrationalStructureMoleculeDriver._A03_REV_EXPECTED
        elif version == "G16RevC.01":
            exp_vals = TestVibrationalStructureMoleculeDriver._C01_REV_EXPECTED
        else:
            self.fail(f"Unknown gaussian version '{version}'")

        return exp_vals

    def setUp(self):
        super().setUp()
        self._molecule = Molecule(
            geometry=[
                ("C", [-0.848629, 2.067624, 0.160992]),
                ("O", [0.098816, 2.655801, -0.159738]),
                ("O", [-1.796073, 1.479446, 0.481721]),
            ],
            multiplicity=1,
            charge=0,
        )

    @data(
        VibrationalStructureDriverType.AUTO,
        VibrationalStructureDriverType.GAUSSIAN_FORCES,
    )
    def test_driver(self, driver_type):
        """test driver"""
        driver = VibrationalStructureMoleculeDriver(
            self._molecule, basis="6-31g", driver_type=driver_type
        )
        try:
            result = driver.run()
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
        self._check_driver_result(self._get_expected_values(), result)

    def _check_driver_result(self, expected_watson_data, watson):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            expected_watson = WatsonHamiltonian(expected_watson_data, 4)
            expected = VibrationalEnergy.from_legacy_driver_result(expected_watson)
        true_vib_energy = cast(VibrationalEnergy, watson.get_property(VibrationalEnergy))

        with self.subTest("one-body terms"):
            expected_one_body = expected.get_vibrational_integral(1)
            true_one_body = true_vib_energy.get_vibrational_integral(1)
            self._check_integrals_are_close(expected_one_body, true_one_body)

        with self.subTest("two-body terms"):
            expected_two_body = expected.get_vibrational_integral(2)
            true_two_body = true_vib_energy.get_vibrational_integral(2)
            self._check_integrals_are_close(expected_two_body, true_two_body)

        with self.subTest("three-body terms"):
            expected_three_body = expected.get_vibrational_integral(3)
            true_three_body = true_vib_energy.get_vibrational_integral(3)
            self._check_integrals_are_close(expected_three_body, true_three_body)

    def _check_integrals_are_close(self, expected, truth):
        for exp, true in zip(expected.integrals, truth.integrals):
            self.assertAlmostEqual(true[0], exp[0])
            self.assertEqual(true[1], exp[1])


if __name__ == "__main__":
    unittest.main()
