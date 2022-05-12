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

"""Test HFInitialPoint."""

import unittest
from unittest.mock import Mock

from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.algorithms.initial_points import HFInitialPoint
from qiskit_nature.circuit.library import UCC
from qiskit_nature.exceptions import QiskitNatureError
from qiskit_nature.properties.second_quantization.second_quantized_property import (
    GroupedSecondQuantizedProperty,
)
from qiskit_nature.properties.second_quantization.electronic.electronic_energy import (
    ElectronicEnergy,
)


class TestHFInitialPoint(QiskitNatureTestCase):
    """Test HFInitialPoint."""

    def setUp(self) -> None:
        super().setUp()
        self.hf_initial_point = HFInitialPoint()
        self.excitation_list = [((0,), (1,))]

    def test_missing_ansatz(self):
        """Test set get ansatz."""
        with self.assertRaises(QiskitNatureError):
            self.hf_initial_point.compute()

    def test_set_get_ansatz(self):
        """Test set get ansatz."""
        ansatz = Mock(spec=UCC)
        ansatz.excitation_list = self.excitation_list
        self.hf_initial_point.ansatz = ansatz
        self.assertEqual(ansatz, self.hf_initial_point.ansatz)
        self.assertEqual(self.excitation_list, self.hf_initial_point.excitation_list)

    def test_set_get_grouped_property(self):
        """Test set get grouped_property."""
        reference_energy = 123.0
        electronic_energy = Mock(spec=ElectronicEnergy)
        electronic_energy.reference_energy = reference_energy
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        grouped_property.get_property = Mock(return_value=electronic_energy)
        self.hf_initial_point.grouped_property = grouped_property
        self.assertEqual(self.hf_initial_point.grouped_property, grouped_property)
        self.assertEqual(self.hf_initial_point._reference_energy, reference_energy)

    def test_set_bad_grouped_property(self):
        """Test set bad grouped_property."""
        grouped_property = Mock()
        self.hf_initial_point.grouped_property = grouped_property
        self.assertEqual(self.hf_initial_point.grouped_property, None)

    def test_set_missing_electronic_energy(self):
        """Test set missing ElectronicEnergy."""
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        grouped_property.get_property = Mock(return_value=None)
        with self.assertWarns(UserWarning):
            self.hf_initial_point.grouped_property = grouped_property
        self.assertEqual(self.hf_initial_point.grouped_property, None)

    def test_set_missing_reference_energy(self):
        """Test set missing reference_energy."""
        electronic_energy = Mock(spec=ElectronicEnergy)
        electronic_energy.reference_energy = None
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        grouped_property.get_property = Mock(return_value=None)
        with self.assertWarns(UserWarning):
            self.hf_initial_point.grouped_property = grouped_property
        self.assertEqual(self.hf_initial_point._reference_energy, 0.0)

    def test_set_bad_ansatz(self):
        """Test set bad ansatz."""
        ansatz = Mock()
        with self.assertRaises(QiskitNatureError):
            self.hf_initial_point.ansatz = ansatz

    def test_set_get_excitation_list(self):
        """Test set get excitation list."""
        self.hf_initial_point.excitation_list = self.excitation_list
        self.assertEqual(self.excitation_list, self.hf_initial_point.excitation_list)

    def test_compute(self):
        """Test length of HF initial point array."""
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        ansatz = Mock(spec=UCC)
        ansatz.excitation_list = self.excitation_list
        self.hf_initial_point.compute(ansatz=ansatz, grouped_property=grouped_property)
        initial_point = self.hf_initial_point.to_numpy_array()
        np.testing.assert_equal(initial_point, np.asarray([0.0]))

    def test_hf_initial_point_is_all_zero(self):
        """Test HF initial point is all zero."""
        self.hf_initial_point.excitation_list = self.excitation_list
        initial_point = self.hf_initial_point.to_numpy_array()
        np.testing.assert_array_equal(initial_point, np.asarray([0.0]))

    def test_hf_energy(self):
        """Test HF energy."""
        reference_energy = 123.0
        electronic_energy = Mock(spec=ElectronicEnergy)
        electronic_energy.reference_energy = reference_energy
        grouped_property = Mock(spec=GroupedSecondQuantizedProperty)
        grouped_property.get_property = Mock(return_value=electronic_energy)
        self.hf_initial_point.grouped_property = grouped_property
        self.hf_initial_point.excitation_list = self.excitation_list
        energy = self.hf_initial_point.get_energy()
        self.assertEqual(energy, 123.0)


if __name__ == "__main__":
    unittest.main()
