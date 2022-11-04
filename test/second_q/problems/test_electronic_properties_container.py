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

"""Tests for the ElectronicPropertiesContainer."""

import unittest

from test import QiskitNatureTestCase

from qiskit_nature.second_q.problems import ElectronicPropertiesContainer
from qiskit_nature.second_q.properties import (
    AngularMomentum,
    ElectronicDensity,
    ElectronicDipoleMoment,
    Magnetization,
    OccupiedModals,
    ParticleNumber,
)


class TestElectronicPropertiesContainer(QiskitNatureTestCase):
    """Tests for the ElectronicPropertiesContainer."""

    def test_angular_momentum(self) -> None:
        """Tests the AngularMomentum property."""
        container = ElectronicPropertiesContainer()

        with self.subTest("initially None"):
            self.assertIsNone(container.angular_momentum)

        with self.subTest("wrong setting type"):
            with self.assertRaises(TypeError):
                container.angular_momentum = OccupiedModals([])  # type: ignore[assignment]

        with self.subTest("successful setting"):
            container.angular_momentum = AngularMomentum(1)
            self.assertIn(AngularMomentum, container)

        with self.subTest("removal via None setting"):
            container.angular_momentum = None
            self.assertNotIn(AngularMomentum, container)

    def test_electronic_density(self) -> None:
        """Tests the ElectronicDensity property."""
        container = ElectronicPropertiesContainer()

        with self.subTest("initially None"):
            self.assertIsNone(container.electronic_density)

        with self.subTest("wrong setting type"):
            with self.assertRaises(TypeError):
                container.electronic_density = OccupiedModals([])  # type: ignore[assignment]

        with self.subTest("successful setting"):
            container.electronic_density = ElectronicDensity()
            self.assertIn(ElectronicDensity, container)

        with self.subTest("removal via None setting"):
            container.electronic_density = None
            self.assertNotIn(ElectronicDensity, container)

    def test_electronic_dipole_moment(self) -> None:
        """Tests the ElectronicDipoleMoment property."""
        container = ElectronicPropertiesContainer()

        with self.subTest("initially None"):
            self.assertIsNone(container.electronic_dipole_moment)

        with self.subTest("wrong setting type"):
            with self.assertRaises(TypeError):
                container.electronic_dipole_moment = OccupiedModals([])  # type: ignore[assignment]

        with self.subTest("successful setting"):
            container.electronic_dipole_moment = ElectronicDipoleMoment(None, None, None)
            self.assertIn(ElectronicDipoleMoment, container)

        with self.subTest("removal via None setting"):
            container.electronic_dipole_moment = None
            self.assertNotIn(ElectronicDipoleMoment, container)

    def test_magnetization(self) -> None:
        """Tests the Magnetization property."""
        container = ElectronicPropertiesContainer()

        with self.subTest("initially None"):
            self.assertIsNone(container.magnetization)

        with self.subTest("wrong setting type"):
            with self.assertRaises(TypeError):
                container.magnetization = OccupiedModals([])  # type: ignore[assignment]

        with self.subTest("successful setting"):
            container.magnetization = Magnetization(1)
            self.assertIn(Magnetization, container)

        with self.subTest("removal via None setting"):
            container.magnetization = None
            self.assertNotIn(Magnetization, container)

    def test_particle_number(self) -> None:
        """Tests the ParticleNumber property."""
        container = ElectronicPropertiesContainer()

        with self.subTest("initially None"):
            self.assertIsNone(container.particle_number)

        with self.subTest("wrong setting type"):
            with self.assertRaises(TypeError):
                container.particle_number = OccupiedModals([])  # type: ignore[assignment]

        with self.subTest("successful setting"):
            container.particle_number = ParticleNumber(1)
            self.assertIn(ParticleNumber, container)

        with self.subTest("removal via None setting"):
            container.particle_number = None
            self.assertNotIn(ParticleNumber, container)

    def test_custom_property(self) -> None:
        """Tests support for custom property objects."""
        container = ElectronicPropertiesContainer()
        container.add(OccupiedModals([]))
        self.assertIn(OccupiedModals, container)


if __name__ == "__main__":
    unittest.main()
