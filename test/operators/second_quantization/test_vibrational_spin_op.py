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
"""Test for VibrationalSpinOp"""

from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt

from qiskit_nature.operators.second_quantization.vibrational_spin_op import VibrationalSpinOp


@ddt
class TestVibrationalSpinOp(QiskitNatureTestCase):
    """VibrationalSpinOp tests."""

    def setUp(self):
        super().setUp()
        self.labels = [("+_1*0 -_1*1", 1215.375), ("+_2*0 -_2*1 +_3*0 -_3*0", -6.385)]
        self.labels_double = [
            ("+_1*0 -_1*1", 2 * 1215.375),
            ("+_2*0 -_2*1 +_3*0 -_3*0", -2 * 6.385),
        ]
        self.labels_divided_3 = [
            ("+_1*0 -_1*1", 1215.375 / 3),
            ("+_2*0 -_2*1 +_3*0 -_3*0", -6.385 / 3),
        ]
        self.labels_neg = [("+_1*0 -_1*1", -1215.375), ("+_2*0 -_2*1 +_3*0 -_3*0", 6.385)]
        self.vibr_spin_op = VibrationalSpinOp(self.labels, 4, 2)

    @staticmethod
    def assertSpinEqual(first: VibrationalSpinOp, second: VibrationalSpinOp):
        """Fail if two VibrationalSpinOps have different matrix representations."""
        np.testing.assert_array_almost_equal(first.to_matrix(), second.to_matrix())

    def test_init_pm_label(self):
        """Test __init__ with plus and minus label"""
        with self.subTest("minus plus"):
            result = VibrationalSpinOp([("+_0*0 -_0*1", 2)], 1, 2)
            desired = [
                ("X_0 X_1", (2 + 0j)),
                ("X_0 Y_1", -2j),
                ("Y_0 X_1", 2j),
                ("Y_0 Y_1", (2 + 0j)),
            ]
            self.assertEqual(result.to_list(), desired)

        with self.subTest("plus minus"):
            result = VibrationalSpinOp([("-_0*0 +_0*1", 2)], 1, 2)
            desired = [
                ("X_0 X_1", (2 + 0j)),
                ("Y_0 X_1", -2j),
                ("X_0 Y_1", 2j),
                ("Y_0 Y_1", (2 + 0j)),
            ]
            self.assertEqual(result.to_list(), desired)

        with self.subTest("plus minus minus plus"):
            result = VibrationalSpinOp([("+_0*0 -_0*1 -_1*0 +_1*1", 3)], 2, 2)
            desired = [
                ("X_0 X_1 X_2 X_3", (3 + 0j)),
                ("X_0 Y_1 X_2 X_3", -3j),
                ("X_0 X_1 Y_2 X_3", -3j),
                ("X_0 Y_1 Y_2 X_3", (-3 - 0j)),
                ("Y_0 X_1 X_2 X_3", 3j),
                ("Y_0 Y_1 X_2 X_3", (3 + 0j)),
                ("Y_0 X_1 Y_2 X_3", (3 + 0j)),
                ("Y_0 Y_1 Y_2 X_3", -3j),
                ("X_0 X_1 X_2 Y_3", 3j),
                ("X_0 Y_1 X_2 Y_3", (3 + 0j)),
                ("X_0 X_1 Y_2 Y_3", (3 + 0j)),
                ("X_0 Y_1 Y_2 Y_3", -3j),
                ("Y_0 X_1 X_2 Y_3", (-3 + 0j)),
                ("Y_0 Y_1 X_2 Y_3", 3j),
                ("Y_0 X_1 Y_2 Y_3", 3j),
                ("Y_0 Y_1 Y_2 Y_3", (3 + 0j)),
            ]

            # Note: the order of list is irrelevant.
            self.assertSetEqual(frozenset(result.to_list()), frozenset(desired))

    @data("X_0*0 +_0*0")
    def test_init_invalid_label(self, label):
        """Test __init__ for invalid label"""
        with self.assertRaises(ValueError):
            VibrationalSpinOp(label, 1, 1)

    def test_neg(self):
        """Test __neg__"""
        actual = -self.vibr_spin_op
        desired = VibrationalSpinOp(self.labels_neg, 4, 2)
        self.assertSpinEqual(actual, desired)

    def test_mul(self):
        """Test __mul__, and __rmul__"""
        actual = self.vibr_spin_op * 2
        desired = VibrationalSpinOp(self.labels_double, 4, 2)
        self.assertSpinEqual(actual, desired)

    def test_div(self):
        """Test __truediv__"""
        actual = self.vibr_spin_op / 3
        desired = VibrationalSpinOp(self.labels_divided_3, 4, 2)
        self.assertSpinEqual(actual, desired)

    def test_add(self):
        """Test __add__"""
        actual = self.vibr_spin_op + self.vibr_spin_op
        desired = VibrationalSpinOp(self.labels_double, 4, 2)
        self.assertSpinEqual(actual, desired)
