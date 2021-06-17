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

""" Test Bravyi-Kitaev Super-Fast Mapper """

import unittest
import numpy as np

from qiskit.quantum_info import SparsePauliOp

from test import QiskitNatureTestCase

from qiskit_nature.mappers.second_quantization import BravyiKitaevSFMapper
from qiskit_nature.mappers.second_quantization.bksf import edge_operator_aij, edge_operator_bi


class TestBravyiKitaevSFMapper(QiskitNatureTestCase):
    """Test Bravyi-Kitaev Super-Fast Mapper"""

    def test_bksf_edge_op_bi(self):
        """Test bksf mapping, edge operator bi"""
        edge_matrix = np.triu(np.ones((4, 4)))
        edge_list = np.array(np.nonzero(np.triu(edge_matrix) - np.diag(np.diag(edge_matrix))))
        qterm_b0 = edge_operator_bi(edge_list, 0)
        qterm_b1 = edge_operator_bi(edge_list, 1)
        qterm_b2 = edge_operator_bi(edge_list, 2)
        qterm_b3 = edge_operator_bi(edge_list, 3)

        ref_qterm_b0 = SparsePauliOp('IIIZZZ')
        ref_qterm_b1 = SparsePauliOp('IZZIIZ')
        ref_qterm_b2 = SparsePauliOp('ZIZIZI')
        ref_qterm_b3 = SparsePauliOp('ZZIZII')

        with self.subTest("Test 1"):
            self.assertEqual(qterm_b0, ref_qterm_b0)
        with self.subTest("Test 2"):
            self.assertEqual(qterm_b1, ref_qterm_b1)
        with self.subTest("Test 3"):
            self.assertEqual(qterm_b2, ref_qterm_b2)
        with self.subTest("Test 4"):
            self.assertEqual(qterm_b3, ref_qterm_b3)

    def test_bksf_edge_op_aij(self):
        """Test bksf mapping, edge operator aij"""
        edge_matrix = np.triu(np.ones((4, 4)))
        edge_list = np.array(np.nonzero(np.triu(edge_matrix) - np.diag(np.diag(edge_matrix))))
        qterm_a01 = edge_operator_aij(edge_list, 0, 1)
        qterm_a02 = edge_operator_aij(edge_list, 0, 2)
        qterm_a03 = edge_operator_aij(edge_list, 0, 3)
        qterm_a12 = edge_operator_aij(edge_list, 1, 2)
        qterm_a13 = edge_operator_aij(edge_list, 1, 3)
        qterm_a23 = edge_operator_aij(edge_list, 2, 3)

        ref_qterm_a01 = SparsePauliOp('IIIIIX')
        ref_qterm_a02 = SparsePauliOp('IIIIXZ')
        ref_qterm_a03 = SparsePauliOp('IIIXZZ')
        ref_qterm_a12 = SparsePauliOp('IIXIZZ')
        ref_qterm_a13 = SparsePauliOp('IXZZIZ')
        ref_qterm_a23 = SparsePauliOp('XZZZZI')

        with self.subTest("Test 1"):
            self.assertEqual(qterm_a01, ref_qterm_a01)
        with self.subTest("Test 2"):
            self.assertEqual(qterm_a02, ref_qterm_a02)
        with self.subTest("Test 3"):
            self.assertEqual(qterm_a03, ref_qterm_a03)
        with self.subTest("Test 4"):
            self.assertEqual(qterm_a12, ref_qterm_a12)
        with self.subTest("Test 5"):
            self.assertEqual(qterm_a13, ref_qterm_a13)
        with self.subTest("Test 6"):
            self.assertEqual(qterm_a23, ref_qterm_a23)


if __name__ == "__main__":
    unittest.main()
