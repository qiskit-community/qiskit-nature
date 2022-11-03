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

"""Tests for the LatticePropertiesContainer."""

import unittest

from test import QiskitNatureTestCase

from qiskit_nature.second_q.problems import LatticePropertiesContainer
from qiskit_nature.second_q.properties import OccupiedModals


class TestLatticePropertiesContainer(QiskitNatureTestCase):
    """Tests for the LatticePropertiesContainer."""

    def test_custom_property(self) -> None:
        """Tests support for custom property objects."""
        container = LatticePropertiesContainer()
        container.add(OccupiedModals([]))
        self.assertIn(OccupiedModals, container)


if __name__ == "__main__":
    unittest.main()
