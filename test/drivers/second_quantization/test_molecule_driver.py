# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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

import unittest
from test import QiskitNatureTestCase, requires_extra_library
from ddt import ddt, data
from qiskit_nature.drivers.second_quantization import (
    MethodType,
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
    VibrationalStructureDriverType,
    VibrationalStructureMoleculeDriver,
)
from qiskit_nature.drivers import Molecule
from qiskit_nature.exceptions import UnsupportMethodError
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy


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

    @requires_extra_library
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
    @requires_extra_library
    def test_driver(self, config):
        """test driver"""
        driver_type, hf_energy = config
        driver = ElectronicStructureMoleculeDriver(
            self._molecule, basis="sto3g", driver_type=driver_type
        )
        driver_result = driver.run()
        electronic_energy = cast(ElectronicEnergy, driver_result.get_property(ElectronicEnergy))
        self.assertAlmostEqual(electronic_energy.reference_energy, hf_energy, places=5)

    @data(
        ElectronicStructureDriverType.PSI4,
        ElectronicStructureDriverType.PYQUANTE,
        ElectronicStructureDriverType.GAUSSIAN,
    )
    @requires_extra_library
    def test_unsupported_method(self, driver_type):
        """test unsupported methods"""
        for method in [MethodType.RKS, MethodType.ROKS, MethodType.UKS]:
            driver = ElectronicStructureMoleculeDriver(
                self._molecule, basis="sto3g", method=method, driver_type=driver_type
            )
            with self.assertRaises(UnsupportMethodError):
                _ = driver.run()


@ddt
class TestVibrationalStructureMoleculeDriver(QiskitNatureTestCase):
    """Vibrational structure Molecule Driver tests."""

    _MOLECULE_EXPECTED = [
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
        [38.72849649956234, 4, 4, 2],
        [38.72849649956234, 3, 3, 2],
        [0.4207357291666667, 2, 2, 2, 2],
        [4.9425425, 1, 1, 2, 2],
        [1.6122932291666665, 1, 1, 1, 1],
        [-4.194299375, 4, 4, 2, 2],
        [-4.194299375, 3, 3, 2, 2],
        [-10.205891875, 4, 4, 1, 1],
        [-10.205891875, 3, 3, 1, 1],
        [1.8255064583333331, 4, 4, 4, 4],
        [3.507156666666667, 4, 4, 4, 3],
        [10.160466875, 4, 4, 3, 3],
        [-3.507156666666667, 4, 3, 3, 3],
        [1.8255065625, 3, 3, 3, 3],
    ]

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

    def _check_driver_result(self, expected, watson):
        for i, entry in enumerate(watson.data):
            msg = f"mode[{i}]={entry} does not match expected {expected[i]}"
            self.assertAlmostEqual(entry[0], expected[i][0], msg=msg)
            self.assertListEqual(entry[1:], expected[i][1:], msg=msg)

    @data(
        VibrationalStructureDriverType.AUTO,
        VibrationalStructureDriverType.GAUSSIAN_FORCES,
    )
    @requires_extra_library
    def test_driver(self, driver_type):
        """test driver"""
        driver = VibrationalStructureMoleculeDriver(
            self._molecule, basis="6-31g", driver_type=driver_type
        )
        result = driver.run()
        self._check_driver_result(TestVibrationalStructureMoleculeDriver._MOLECULE_EXPECTED, result)


if __name__ == "__main__":
    unittest.main()
