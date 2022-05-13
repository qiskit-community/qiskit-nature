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

"""Test VSCFInitialPoint."""

import unittest
from unittest.mock import Mock

from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.algorithms.initial_points import VSCFInitialPoint
from qiskit_nature.circuit.library import UVCC
from qiskit_nature.exceptions import QiskitNatureError


class TestVSCFInitialPoint(QiskitNatureTestCase):
    """Test VSCFInitialPoint."""

    def setUp(self) -> None:
        super().setUp()
        self.vscf_initial_point = VSCFInitialPoint()
        self.excitation_list = [((0,), (1,))]

    def test_missing_ansatz(self):
        """Test set get ansatz."""
        with self.assertRaises(QiskitNatureError):
            self.vscf_initial_point.compute()

    def test_set_get_ansatz(self):
        """Test set get ansatz."""
        ansatz = Mock(spec=UVCC)
        ansatz.excitation_list = self.excitation_list
        self.vscf_initial_point.ansatz = ansatz
        self.assertEqual(ansatz, self.vscf_initial_point.ansatz)
        self.assertEqual(self.excitation_list, self.vscf_initial_point.excitation_list)

    def test_vscf_initial_point_is_all_zero(self):
        """Test VSCF initial point is all zero."""
        self.vscf_initial_point.excitation_list = self.excitation_list
        initial_point = self.vscf_initial_point.to_numpy_array()
        np.testing.assert_array_equal(initial_point, np.asarray([0.0]))


if __name__ == "__main__":
    unittest.main()
