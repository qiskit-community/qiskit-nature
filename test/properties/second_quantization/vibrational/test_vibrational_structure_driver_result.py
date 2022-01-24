# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test VibrationalStructureDriverResult Property"""

from test import QiskitNatureTestCase

import h5py
import numpy as np

from qiskit_nature.drivers.second_quantization import GaussianForcesDriver
from qiskit_nature.properties.second_quantization.vibrational import (
    OccupiedModals,
    VibrationalEnergy,
    VibrationalStructureDriverResult,
)
from qiskit_nature.properties.second_quantization.vibrational.bases import HarmonicBasis
from qiskit_nature.properties.second_quantization.vibrational.integrals import VibrationalIntegrals


class TestVibrationalStructureDriverResult(QiskitNatureTestCase):
    """Test VibrationalStructureDriverResult Property"""

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

    def compare_occupied_modals(
        self, first: OccupiedModals, second: OccupiedModals, msg: str = None
    ) -> None:
        # pylint: disable=unused-argument
        """Compares two OccupiedModals instances."""
        pass

    def setUp(self) -> None:
        """Setup expected object."""
        super().setUp()
        self.addTypeEqualityFunc(VibrationalIntegrals, self.compare_vibrational_integral)
        self.addTypeEqualityFunc(VibrationalEnergy, self.compare_vibrational_energy)
        self.addTypeEqualityFunc(OccupiedModals, self.compare_occupied_modals)

        driver = GaussianForcesDriver(
            logfile=self.get_resource_path(
                "test_driver_gaussian_log.txt", "drivers/second_quantization/gaussiand"
            )
        )
        self.expected = driver.run()
        self.expected.basis = HarmonicBasis([3])

    def test_from_hdf5(self):
        """Test from_hdf5."""
        with h5py.File(
            self.get_resource_path(
                "vibrational_structure_driver_result.hdf5",
                "properties/second_quantization/vibrational/resources",
            ),
            "r",
        ) as file:
            for group in file.values():
                prop = VibrationalStructureDriverResult.from_hdf5(group)
                for inner_prop in iter(prop):
                    expected = self.expected.get_property(type(inner_prop))
                    self.assertEqual(inner_prop, expected)
