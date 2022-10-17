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

"""PropertyTest class"""

from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.second_q.hamiltonians import VibrationalEnergy
from qiskit_nature.second_q.properties import (
    AngularMomentum,
    Magnetization,
    ParticleNumber,
)
from qiskit_nature.second_q.properties import OccupiedModals
from qiskit_nature.second_q.properties.integrals import VibrationalIntegrals


class PropertyTest(QiskitNatureTestCase):
    """Property instance tester"""

    def compare_angular_momentum(
        self, first: AngularMomentum, second: AngularMomentum, msg: str = None
    ) -> None:
        """Compares two AngularMomentum instances."""
        if first.num_spatial_orbitals != second.num_spatial_orbitals:
            raise self.failureException(msg)

    def compare_magnetization(
        self, first: Magnetization, second: Magnetization, msg: str = None
    ) -> None:
        """Compares two Magnetization instances."""
        if first.num_spatial_orbitals != second.num_spatial_orbitals:
            raise self.failureException(msg)

    def compare_particle_number(
        self, first: ParticleNumber, second: ParticleNumber, msg: str = None
    ) -> None:
        """Compares two ParticleNumber instances."""
        if first.num_spatial_orbitals != second.num_spatial_orbitals:
            raise self.failureException(msg)

    def compare_vibrational_integral(
        self, first: VibrationalIntegrals, second: VibrationalIntegrals, msg: str = None
    ) -> None:
        """Compares two VibrationalIntegral instances."""
        if first.name != second.name:
            raise self.failureException(msg)

        if first.num_body_terms != second.num_body_terms:
            raise self.failureException(msg)

        for f_int, s_int in zip(first.integrals, second.integrals):
            if not np.isclose(f_int[0], s_int[0]):
                raise self.failureException(msg)

            if not all(f == s for f, s in zip(f_int[1:], s_int[1:])):
                raise self.failureException(msg)

    def compare_vibrational_energy(
        self, first: VibrationalEnergy, second: VibrationalEnergy, msg: str = None
    ) -> None:
        # pylint: disable=unused-argument
        """Compares two VibrationalEnergy instances."""
        for f_ints, s_ints in zip(first, second):
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
        self.addTypeEqualityFunc(AngularMomentum, self.compare_angular_momentum)
        self.addTypeEqualityFunc(Magnetization, self.compare_magnetization)
        self.addTypeEqualityFunc(ParticleNumber, self.compare_particle_number)
        self.addTypeEqualityFunc(VibrationalIntegrals, self.compare_vibrational_integral)
        self.addTypeEqualityFunc(VibrationalEnergy, self.compare_vibrational_energy)
        self.addTypeEqualityFunc(OccupiedModals, self.compare_occupied_modals)
