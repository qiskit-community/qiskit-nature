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
from qiskit_nature.properties.second_quantization.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)


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

    def test_set_grouped_property(self):
        """Test set get grouped_property (not used for VSCF)."""
        self.assertIsNone(self.vscf_initial_point.grouped_property)
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        self.vscf_initial_point.grouped_property = grouped_property
        self.assertEqual(grouped_property, self.vscf_initial_point.grouped_property)

    def test_vscf_with_excitation_list_set_directly(self):
        """Test VSCF initial point is all zero when called on demand."""
        self.vscf_initial_point.excitation_list = self.excitation_list
        initial_point = self.vscf_initial_point.to_numpy_array()
        np.testing.assert_array_equal(initial_point, np.asarray([0.0]))

    def test_vscf_compute(self):
        """Test VSCF initial point is all zero when called via compute."""
        ansatz = Mock(spec=UVCC)
        ansatz.excitation_list = self.excitation_list
        self.vscf_initial_point.compute(ansatz)
        initial_point = self.vscf_initial_point.to_numpy_array()
        np.testing.assert_array_equal(initial_point, np.asarray([0.0]))


if __name__ == "__main__":
    unittest.main()
