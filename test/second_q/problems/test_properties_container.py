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

"""Tests for the PropertiesContainer."""

import unittest

from test import QiskitNatureTestCase

from qiskit_nature.second_q.problems import PropertiesContainer
from qiskit_nature.second_q.properties import OccupiedModals


class TestPropertiesContainer(QiskitNatureTestCase):
    """Tests for the PropertiesContainer."""

    def setUp(self) -> None:
        super().setUp()
        self.property = OccupiedModals([])

    def test_contains(self) -> None:
        """Tests the `__contains__` method."""
        container = PropertiesContainer()
        container.add(self.property)

        with self.subTest("object check"):
            self.assertIn(self.property, container)

        with self.subTest("type check"):
            self.assertIn(OccupiedModals, container)

        with self.subTest("str check"):
            self.assertIn("OccupiedModals", container)

        with self.subTest("TypeError"):
            with self.assertRaises(TypeError):
                self.assertIn(1, container)

    def test_add(self) -> None:
        """Tests the `add` method."""
        with self.subTest("Successful addition"):
            container = PropertiesContainer()
            container.add(self.property)
            self.assertIn(self.property, container)

        with self.subTest("Unsuccessful addition if object already exists"):
            container = PropertiesContainer()
            container.add(self.property)
            with self.assertRaises(ValueError):
                container.add(self.property)

    def test_discard(self) -> None:
        """Tests the `discard` method."""
        with self.subTest("Discard via object"):
            container = PropertiesContainer()
            container.add(self.property)
            container.discard(self.property)
            self.assertNotIn(self.property, container)

        with self.subTest("Discard via type"):
            container = PropertiesContainer()
            container.add(self.property)
            container.discard(OccupiedModals)
            self.assertNotIn(OccupiedModals, container)

        with self.subTest("Discard via str"):
            container = PropertiesContainer()
            container.add(self.property)
            container.discard("OccupiedModals")
            self.assertNotIn("OccupiedModals", container)

        with self.subTest("TypeError"):
            container = PropertiesContainer()
            container.add(self.property)
            with self.assertRaises(TypeError):
                container.discard(1)  # type: ignore[arg-type]

    def test_remove(self) -> None:
        """Tests the `remove` method."""
        with self.subTest("Remove via object"):
            container = PropertiesContainer()
            container.add(self.property)
            container.remove(self.property)
            self.assertNotIn(self.property, container)

        with self.subTest("Remove via type"):
            container = PropertiesContainer()
            container.add(self.property)
            container.remove(OccupiedModals)
            self.assertNotIn(OccupiedModals, container)

        with self.subTest("Remove via str"):
            container = PropertiesContainer()
            container.add(self.property)
            container.remove("OccupiedModals")
            self.assertNotIn("OccupiedModals", container)

        with self.subTest("TypeError"):
            container = PropertiesContainer()
            container.add(self.property)
            with self.assertRaises(TypeError):
                container.remove(1)

        with self.subTest("KeyError"):
            container = PropertiesContainer()
            container.add(self.property)
            with self.assertRaises(KeyError):
                container.remove("MissingProperty")

    def test_pop(self) -> None:
        """Tests the `pop` method."""
        with self.subTest("Pop via object"):
            container = PropertiesContainer()
            container.add(self.property)
            prop = container.pop()
            self.assertNotIn(self.property, container)
            self.assertEqual(prop, self.property)

        with self.subTest("Pop via type"):
            container = PropertiesContainer()
            container.add(self.property)
            prop = container.pop()
            self.assertNotIn(OccupiedModals, container)
            self.assertEqual(prop, self.property)

        with self.subTest("Pop via str"):
            container = PropertiesContainer()
            container.add(self.property)
            prop = container.pop()
            self.assertNotIn("OccupiedModals", container)
            self.assertEqual(prop, self.property)

        with self.subTest("KeyError"):
            container = PropertiesContainer()
            with self.assertRaises(KeyError):
                _ = container.pop()

    def test_clear(self) -> None:
        """Tests the `clear` method."""
        container = PropertiesContainer()
        container.add(self.property)
        container.clear()
        self.assertNotIn(self.property, container)

    def test_len(self) -> None:
        """Tests the `len` method."""
        container = PropertiesContainer()
        self.assertEqual(len(container), 0)
        container.add(self.property)
        self.assertEqual(len(container), 1)

    def test_iter(self) -> None:
        """Tests the `iter` method."""
        container = PropertiesContainer()
        container.add(self.property)
        properties = list(container)
        self.assertEqual(properties, [self.property])


if __name__ == "__main__":
    unittest.main()
