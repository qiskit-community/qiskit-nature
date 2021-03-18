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
from ddt import ddt, data, unpack
from qiskit_nature.problems.second_quantization.vibrational.vibr_to_spin_op_label_converter import \
    convert_to_spin_op_labels


@ddt
class TestVibrToSpinOpLabelConverter(QiskitNatureTestCase):
    """Tests Vibrational Spin Op to Spin Op Label Converter."""

    test_cases_correct = [
        [[('-_0*1 +_0*0', 1215.375), ('-_2*1 +_2*0 -_0*1 +_0*0', -6.385)], 3, 2,
         [('-_1 +_0', 1215.375), ('-_5 +_4 -_1 +_0', -6.385)]],
        [[('-_0*1 +_0*0', 1215.375), ('-_2*1 +_2*0 -_1*1 +_1*0', -6.385)], 3, [2, 2, 2],
         [('-_1 +_0', 1215.375), ('-_5 +_4 -_3 +_2', -6.385)]]
    ]

    part_num_not_conserved = [
        [('-_1*1 +_0*0', 1215.375), ('-_2*1 +_2*0 -_1*0 +_0*0', -6.385)], 3, [2, 2, 2]]
    mode_out_of_range = [[('-_1*1 +_1*0', 1215.375), ('+_3*0 -_3*0 -_2*1 +_2*0', -6.385)], 3, 2]
    modal_out_of_range = [[('-_0*2 +_0*0', 1215.375), ('-_2*1 +_2*0 +_1*0 -_1*0', -6.385)], 3, 2]
    mode_order_incorrect = [[('+_0*0 -_0*0', 1215.375), ('+_2*0 +_1*0 -_2*1 -_1*0', -6.385)], 3, 2]
    modal_order_incorrect = [[('+_0*0 -_0*0', 1215.375), ('+_2*0 -_2*1 +_1*0 -_1*0', -6.385)], 3, 2]
    ops_order_incorrect = [[('-_0*0 +_0*0', 1215.375), ('+_2*1 -_2*1 +_1*0 -_1*0', -6.385)], 3, 2]
    duplicate_terms_consecutive = [
        [('+_0*0 -_0*0', 1215.375), ('-_2*1 +_2*0 +_1*0 +_1*0 -_1*0 -_1*0', -6.385)],
        3, 2]

    test_cases_with_exceptions = [
        part_num_not_conserved,
        mode_out_of_range,
        modal_out_of_range,
        mode_order_incorrect,
        modal_order_incorrect,
        ops_order_incorrect,
        duplicate_terms_consecutive,
    ]

    @data(*test_cases_correct)
    @unpack
    def test_convert_to_spin_op_labels(self, vibrational_labels, num_modes, num_modals, expected):
        """Tests that VibrationalSpinOp labels are converted to SpinOp labels correctly."""

        spin_op_labels = convert_to_spin_op_labels(vibrational_labels, num_modes, num_modals)

        self.assertListEqual(spin_op_labels, expected)

    @data(*test_cases_with_exceptions)
    @unpack
    def test_convert_to_spin_op_labels_invalid_labels(self, vibrational_labels, num_modes,
                                                      num_modals):
        """Tests that VibrationalSpinOp to SpinOp labels converter throws an exception when
        provided with invalid labels."""
        self.assertRaises(ValueError, convert_to_spin_op_labels, vibrational_labels, num_modes,
                          num_modals)
