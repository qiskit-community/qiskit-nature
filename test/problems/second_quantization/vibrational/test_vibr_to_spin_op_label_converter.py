# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests Vibrational Spin Op to Spin Op Label Converter."""
from test import QiskitNatureTestCase
from qiskit_nature.problems.second_quantization.vibrational.vibr_to_spin_op_label_converter import \
    convert_to_spin_op_labels


class TestVibrToSpinOpLabelConverter(QiskitNatureTestCase):
    """Tests Vibrational Spin Op to Spin Op Label Converter."""

    def test_convert_to_spin_op_labels(self):
        """Tests that VibrationalSpinOp labels are converted to SpinOp labels correctly."""
        expected_spin_op_labels = [('+_0 -_1', 1215.375), ('+_4 +_0 -_5 -_1', -6.385)]
        vibrational_labels = [('+_0*0 -_0*1', 1215.375), ('+_2*0 +_0*0 -_2*1 -_0*1', -6.385)]
        num_modes = 3
        num_modals = 2

        spin_op_labels = convert_to_spin_op_labels(vibrational_labels, num_modes, num_modals)

        self.assertListEqual(spin_op_labels, expected_spin_op_labels)

    def test_convert_to_spin_op_labels_listed_modals(self):
        """Tests that VibrationalSpinOp labels are converted to SpinOp labels correctly when
        number of modals are provided as a list."""
        expected_spin_op_labels = [('+_0 -_1', 1215.375), ('+_4 +_2 -_5 -_3', -6.385)]
        vibrational_labels = [('+_0*0 -_0*1', 1215.375), ('+_2*0 +_1*0 -_2*1 -_1*1', -6.385)]
        num_modes = 3
        num_modals = [2, 2, 2]

        spin_op_labels = convert_to_spin_op_labels(vibrational_labels, num_modes, num_modals)

        self.assertListEqual(spin_op_labels, expected_spin_op_labels)

    def test_convert_to_spin_op_labels_modes_disagree(self):
        """Tests that VibrationalSpinOp to SpinOp labels converter throws an exception when
        indices of modes in a label do not agree between raising and lowering operators."""
        vibrational_labels = [('+_0*0 -_1*1', 1215.375), ('+_2*0 +_0*0 -_2*1 -_1*0', -6.385)]
        num_modes = 3
        num_modals = [2, 2, 2]

        self.assertRaises(ValueError, convert_to_spin_op_labels, vibrational_labels, num_modes,
                          num_modals)

    def test_convert_to_spin_op_labels_bad_mode(self):
        """Tests that VibrationalSpinOp to SpinOp labels converter throws an exception when a
        mode index is out of range."""
        vibrational_labels = [('+_0*0 -_1*1', 1215.375), ('+_2*0 +_3*0 -_2*1 -_1*0', -6.385)]
        num_modes = 3
        num_modals = 2

        self.assertRaises(ValueError, convert_to_spin_op_labels, vibrational_labels, num_modes,
                          num_modals)

    def test_convert_to_spin_op_labels_bad_modal(self):
        """Tests that VibrationalSpinOp to SpinOp labels converter throws an exception when a
        modal index is out of range."""
        vibrational_labels = [('+_0*0 -_1*2', 1215.375), ('+_2*0 +_1*0 -_2*1 -_1*0', -6.385)]
        num_modes = 3
        num_modals = 2

        self.assertRaises(ValueError, convert_to_spin_op_labels, vibrational_labels, num_modes,
                          num_modals)
