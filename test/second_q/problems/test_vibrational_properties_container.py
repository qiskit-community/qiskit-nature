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

"""Tests for the VibrationalPropertiesContainer."""

import unittest

from test import QiskitNatureTestCase

from qiskit_nature.second_q.problems import VibrationalPropertiesContainer
from qiskit_nature.second_q.properties import Magnetization, OccupiedModals


class TestVibrationalPropertiesContainer(QiskitNatureTestCase):
    """Tests for the VibrationalPropertiesContainer."""

    def test_occupied_modals(self) -> None:
        """Tests the OccupiedModals property."""
        container = VibrationalPropertiesContainer()

        with self.subTest("initially None"):
            self.assertIsNone(container.occupied_modals)

        with self.subTest("wrong setting type"):
            with self.assertRaises(TypeError):
                container.occupied_modals = Magnetization(2)  # type: ignore[assignment]

        with self.subTest("successful setting"):
            container.occupied_modals = OccupiedModals([])
            self.assertIn(OccupiedModals, container)

        with self.subTest("removal via None setting"):
            container.occupied_modals = None
            self.assertNotIn(OccupiedModals, container)

    def test_custom_property(self) -> None:
        """Tests support for custom property objects."""
        container = VibrationalPropertiesContainer()
        container.add(Magnetization(2))
        self.assertIn(Magnetization, container)


if __name__ == "__main__":
    unittest.main()
