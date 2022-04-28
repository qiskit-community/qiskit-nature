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

"""Test the initial point generator base class."""

import unittest
from unittest.mock import patch
from test import QiskitNatureTestCase

from qiskit_nature.algorithms import InitialPoint


class TestInitialPoint(QiskitNatureTestCase):
    @patch.multiple(InitialPoint, __abstractmethods__=set())
    def test_initial_point_raises_not_implemented_error(self):
        """Test all methods and properties are not implemented."""
        initial_point = InitialPoint()
        with self.subTest("compute") and self.assertRaises(NotImplementedError):
            initial_point.compute(None, None)

        with self.subTest("to_numpy_array") and self.assertRaises(NotImplementedError):
            initial_point.to_numpy_array()

        with self.subTest("get grouped_property") and self.assertRaises(NotImplementedError):
            _ = initial_point.grouped_property

        with self.subTest("set grouped_property") and self.assertRaises(NotImplementedError):
            initial_point.grouped_property = None

        with self.subTest("get ansatz") and self.assertRaises(NotImplementedError):
            _ = initial_point.ansatz

        with self.subTest("set ansatz") and self.assertRaises(NotImplementedError):
            initial_point.ansatz = None


if __name__ == "__main__":
    unittest.main()
