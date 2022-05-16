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

# pylint: disable=line-too-long
from qiskit_nature.properties.second_quantization.electronic.integrals.electronic_integrals_utils.two_body_symmetry_conversion import (
    phys_to_chem,
    chem_to_phys,
    find_index_order,
    IndexType,
)


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

    def test_phys_to_chem(self):
        """Test correct conversion to chemists' index order"""
        # expected = np.einsum("ijkl->iljk", self.TWO_BODY_PHYS)
        expected = self.TWO_BODY_CHEM
        actual = phys_to_chem(self.TWO_BODY_PHYS)
        self.assertTrue(np.allclose(expected, actual))

    def test_chem_to_phys(self):
        """Test correctly conversion to physicists' index order"""
        # expected = np.einsum("pqrs->prsq", self.TWO_BODY_CHEM)
        expected = self.TWO_BODY_PHYS
        actual = chem_to_phys(self.TWO_BODY_CHEM)
        self.assertTrue(np.allclose(expected, actual))

    def test_find_index_order_chem(self):
        """Test correctly identifies chemists' index order"""
        result = find_index_order(self.TWO_BODY_CHEM)
        self.assertEqual(result, IndexType.CHEM)

    def test_find_index_order_phys(self):
        """Test correctly identifies physicists' index order"""
        result = find_index_order(self.TWO_BODY_PHYS)
        self.assertEqual(result, IndexType.PHYS)

    def test_find_index_order_intermediate(self):
        """Test correctly identifies intermediate index order"""

    def test_find_index_order_unknown(self):
        """Test correct return for unknown index order"""
