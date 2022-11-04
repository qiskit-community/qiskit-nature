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

from itertools import zip_longest

from qiskit_nature.second_q.properties import (
    AngularMomentum,
    Magnetization,
    ParticleNumber,
    OccupiedModals,
)


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

    def compare_occupied_modals(
        self, first: OccupiedModals, second: OccupiedModals, msg: str = None
    ) -> None:
        # pylint: disable=unused-argument
        """Compares two OccupiedModals instances."""
        if any(f != s for f, s in zip_longest(first.num_modals, second.num_modals)):
            raise self.failureException(msg)

    def setUp(self) -> None:
        """Setup expected object."""
        super().setUp()
        self.addTypeEqualityFunc(AngularMomentum, self.compare_angular_momentum)
        self.addTypeEqualityFunc(Magnetization, self.compare_magnetization)
        self.addTypeEqualityFunc(ParticleNumber, self.compare_particle_number)
        self.addTypeEqualityFunc(OccupiedModals, self.compare_occupied_modals)
