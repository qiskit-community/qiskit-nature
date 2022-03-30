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

"""Test the point generator base class."""

import unittest
from unittest.mock import patch
from test import QiskitNatureTestCase

from qiskit_nature.algorithms import InitialPoint


class TestPointGenerator(QiskitNatureTestCase):
    @patch.multiple(InitialPoint, __abstractmethods__=set())
    def test_initial_point_raises_not_implemented_error(self):
        initial_point = InitialPoint()
        with self.assertRaises(NotImplementedError):
            _ = initial_point.get_initial_point(None, None)


if __name__ == "__main__":
    unittest.main()
