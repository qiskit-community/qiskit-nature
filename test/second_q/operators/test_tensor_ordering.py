# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test two body symmetry conversion utils"""

from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt, unpack

from qiskit_nature.second_q.operators.tensor_ordering import (
    to_chemist_ordering,
    to_physicist_ordering,
    find_index_order,
    IndexType,
)
from qiskit_nature.exceptions import QiskitNatureError


@ddt
class TestTwoBodySymmetryConversion(QiskitNatureTestCase):
    """Tests for two body symmetry conversion utils"""

    TWO_BODY_CHEM = np.asarray(
        [
            [[[0.67571015, 0.0], [0.0, 0.66458173]], [[0.0, 0.1809312], [0.1809312, 0.0]]],
            [[[0.0, 0.1809312], [0.1809312, 0.0]], [[0.66458173, 0.0], [0.0, 0.69857372]]],
        ]
    )

    TWO_BODY_PHYS = np.asarray(
        [
            [[[0.67571015, 0.0], [0.0, 0.1809312]], [[0.0, 0.1809312], [0.66458173, 0.0]]],
            [[[0.0, 0.66458173], [0.1809312, 0.0]], [[0.1809312, 0.0], [0.0, 0.69857372]]],
        ]
    )

    TWO_BODY_INTERMEDIATE = np.asarray(
        [
            [[[0.67571015, 0.0], [0.0, 0.1809312]], [[0.0, 0.66458173], [0.1809312, 0.0]]],
            [[[0.0, 0.1809312], [0.66458173, 0.0]], [[0.1809312, 0.0], [0.0, 0.69857372]]],
        ]
    )

    TWO_BODY_UNKNOWN = np.asarray(
        [
            [[[0.67571015, 0.0], [0.0, 0.1809312]], [[0.0, 0.0], [0.1809312, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.69857372]]],
        ]
    )

    @unpack
    @data(
        (TWO_BODY_CHEM, TWO_BODY_PHYS),  # test chem to phys
        (TWO_BODY_PHYS, TWO_BODY_PHYS),  # test phys to phys
        (TWO_BODY_INTERMEDIATE, TWO_BODY_PHYS),  # test intermediate to phys
    )
    def test_to_physicist_ordering(self, initial, expected):
        """Test correct conversion to physicists' index order"""
        actual = to_physicist_ordering(initial)
        self.assertTrue(np.allclose(expected, actual))

    def test_unknown_to_physicist_ordering(self):
        """Test to_physicist_ordering raises exception with unknown index input"""
        with self.assertRaises(QiskitNatureError):
            to_physicist_ordering(self.TWO_BODY_UNKNOWN)

    @unpack
    @data(
        (TWO_BODY_PHYS, TWO_BODY_CHEM),  # test phys to chem
        (TWO_BODY_CHEM, TWO_BODY_CHEM),  # test chem to chem
        (TWO_BODY_INTERMEDIATE, TWO_BODY_CHEM),  # test intermediate to chem
    )
    def test_to_chemist_ordering(self, initial, expected):
        """Test correct conversion to chemists' index order"""
        actual = to_chemist_ordering(initial)
        self.assertTrue(np.allclose(expected, actual))

    def test_unknown_to_chemist_ordering(self):
        """Test to_chemist_ordering raises exception with unknown index input"""
        with self.assertRaises(QiskitNatureError):
            to_chemist_ordering(self.TWO_BODY_UNKNOWN)

    @unpack
    @data(
        (TWO_BODY_PHYS, IndexType.PHYSICIST),
        (TWO_BODY_CHEM, IndexType.CHEMIST),
        (TWO_BODY_INTERMEDIATE, IndexType.INTERMEDIATE),
        (TWO_BODY_UNKNOWN, IndexType.UNKNOWN),
    )
    def test_find_index_order(self, initial, expected):
        """Test correctly identifies index order"""
        result = find_index_order(initial)
        self.assertEqual(result, expected)
