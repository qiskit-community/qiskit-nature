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

"""Test InitialPoint"""

import unittest
from unittest.mock import patch
from test import QiskitNatureTestCase

from qiskit_nature.second_q.algorithms.initial_points import InitialPoint


class TestInitialPoint(QiskitNatureTestCase):
    """Test Initial Point"""

    @patch.multiple(InitialPoint, __abstractmethods__=set())
    def setUp(self) -> None:
        super().setUp()
        # pylint: disable=abstract-class-instantiated
        self.initial_point = InitialPoint()  # type: ignore

    def test_to_numpy_array(self):
        """Test to_numpy_array."""
        with self.assertRaises(NotImplementedError):
            self.initial_point.to_numpy_array()

    def test_get_ansatz(self):
        """Test get ansatz."""
        with self.assertRaises(NotImplementedError):
            _ = self.initial_point.ansatz

    def test_set_ansatz(self):
        """Test set ansatz."""
        with self.assertRaises(NotImplementedError):
            self.initial_point.ansatz = None

    def test_get_grouped_property(self):
        """Test get grouped_property."""
        with self.assertRaises(NotImplementedError):
            _ = self.initial_point.grouped_property

    def test_set_grouped_property(self):
        """Test set grouped_property."""
        with self.assertRaises(NotImplementedError):
            self.initial_point.grouped_property = None

    def test_compute(self):
        """Test compute."""
        with self.assertRaises(NotImplementedError):
            self.initial_point.compute(None, None)


if __name__ == "__main__":
    unittest.main()
