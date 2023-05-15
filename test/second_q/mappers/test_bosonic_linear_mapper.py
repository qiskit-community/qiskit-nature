# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Bosonic Linear Mapper """

import unittest

from test import QiskitNatureTestCase

from ddt import ddt, data, unpack
import numpy as np

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import BosonicOp
from qiskit_nature.second_q.mappers import BosonicLinearMapper
from qiskit_nature import settings


@ddt
class TestBosonicLinearMapper(QiskitNatureTestCase):
    """Test Bosonic Linear Mapper"""

    # Define some useful coefficients
    sq_2 = np.sqrt(2)

    bos_op1 = BosonicOp({"+_0": 1})
    # Using: truncation = 1
    ref_qubit_op1_tr1 = SparsePauliOp(["XX", "YY", "XY", "YX"], coeffs=[0.25, 0.25, -0.25j, 0.25j])
    # Using: truncation = 2
    ref_qubit_op1_tr2 = SparsePauliOp(["IXX", "IYY", "IXY", "IYX", "XXI", "YYI", "XYI", "YXI"],
                                      coeffs=[0.25, 0.25, -0.25j, 0.25j,
                                              sq_2/4, sq_2/4, -1j*sq_2/4, 1j*sq_2/4])

    bos_op2 = BosonicOp({"-_0": 1})
    # Using: truncation = 1
    ref_qubit_op2_tr1 = SparsePauliOp(["XX", "YY", "XY", "YX"], coeffs=[0.25, 0.25, 0.25j, -0.25j])
    # Using: truncation = 2
    ref_qubit_op2_tr2 = SparsePauliOp(["IXX", "IYY", "IXY", "IYX", "XXI", "YYI", "XYI", "YXI"],
                                      coeffs=[0.25, 0.25, 0.25j, -0.25j,
                                              sq_2/4, sq_2/4, 1j*sq_2/4, -1j*sq_2/4])

    bos_op3 = BosonicOp({"+_1": 1})
    # Using: truncation = 1
    ref_qubit_op3_tr1 = SparsePauliOp(["XXII", "YYII", "XYII", "YXII"],
                                      coeffs=[0.25, 0.25, -0.25j, 0.25j])
    # Using: truncation = 2
    ref_qubit_op3_tr2 = SparsePauliOp(
        ["IXXIII", "IYYIII", "IXYIII", "IYXIII", "XXIIII", "YYIIII", "XYIIII", "YXIIII"],
        coeffs=[0.25, 0.25, -0.25j, 0.25j, sq_2/4, sq_2/4, -1j*sq_2/4, 1j*sq_2/4]
    )

    bos_op4 = BosonicOp({"-_1": 1})
    # Using: truncation = 1
    ref_qubit_op4_tr1 = SparsePauliOp(["XXII", "YYII", "XYII", "YXII"],
                                      coeffs=[0.25, 0.25, 0.25j, -0.25j])
    # Using: truncation = 2
    ref_qubit_op4_tr2 = SparsePauliOp(
        ["IXXIII", "IYYIII", "IXYIII", "IYXIII", "XXIIII", "YYIIII", "XYIIII", "YXIIII"],
        coeffs=[0.25, 0.25, 0.25j, -0.25j, sq_2/4, sq_2/4, 1j*sq_2/4, -1j*sq_2/4]
    )

    bos_op5 = BosonicOp({"+_0 -_0": 1})
    # Using: truncation = 1
    ref_qubit_op5_tr1 = SparsePauliOp(["II", "ZZ", "IZ", "ZI"], coeffs=[0.25, -0.25, -0.25, 0.25])
    # Using: truncation = 2
    ref_qubit_op5_tr2 = SparsePauliOp(["III", "IZZ", "IIZ", "IZI", "ZZI", "ZII"],
                                      coeffs=[0.75, -0.25, -0.25, -0.25, -0.5, 0.5])

    bos_op6 = BosonicOp({"-_0 +_0": 1})  # TODO: Check
    # Using: truncation = 1
    ref_qubit_op6_tr1 = SparsePauliOp(["II", "ZZ", "IZ", "ZI"], coeffs=[0.25, -0.25, +0.25, -0.25])
    # Using: truncation = 2
    ref_qubit_op6_tr2 = SparsePauliOp(["III", "IZZ", "IIZ", "IZI", "ZZI", "ZII"],
                                      coeffs=[0.75, -0.25, -0.25, -0.25, -0.5, 0.5])

    bos_op7 = BosonicOp({"+_0 -_1": 1})
    bos_op8 = BosonicOp({"-_1 +_0": 1})
    # Using: truncation = 1
    ref_qubit_op7_8_tr1 = SparsePauliOp(["XXXX", "XXYY", "XXXY", "XXYX",
                                         "YYXX", "YYYY", "YYXY", "YYYX",
                                         "XYXX", "XYYY", "XYXY", "XYYX",
                                         "YXXX", "YXYY", "YXXY", "YXYX"],
                                        coeffs=[1/16, 1/16, -1j/16, 1j/16,
                                                1/16, 1/16, -1j/16, 1j/16,
                                                1j/16, 1j/16, 1/16, -1/16,
                                                -1j/16, -1j/16, -1/16, 1/16])
    # Using: truncation = 2
    ref_qubit_op7_8_tr2 = SparsePauliOp(
        ["IXXIXX", "IXXIYY", "IXXIXY", "IXXIYX", "IXXXXI", "IXXYYI", "IXXXYI", "IXXYXI",
         "IYYIXX", "IYYIYY", "IYYIXY", "IYYIYX", "IYYXXI", "IYYYYI", "IYYXYI", "IYYYXI",
         "IXYIXX", "IXYIYY", "IXYIXY", "IXYIYX", "IXYXXI", "IXYYYI", "IXYXYI", "IXYYXI",
         "IYXIXX", "IYXIYY", "IYXIXY", "IYXIYX", "IYXXXI", "IYXYYI", "IYXXYI", "IYXYXI",
         "XXIIXX", "XXIIYY", "XXIIXY", "XXIIYX", "XXIXXI", "XXIYYI", "XXIXYI", "XXIYXI",
         "YYIIXX", "YYIIYY", "YYIIXY", "YYIIYX", "YYIXXI", "YYIYYI", "YYIXYI", "YYIYXI",
         "XYIIXX", "XYIIYY", "XYIIXY", "XYIIYX", "XYIXXI", "XYIYYI", "XYIXYI", "XYIYXI",
         "YXIIXX", "YXIIYY", "YXIIXY", "YXIIYX", "YXIXXI", "YXIYYI", "YXIXYI", "YXIYXI"],
        coeffs=[1/16, 1/16, -1j/16, 1j/16, sq_2/16, sq_2/16, -sq_2*1j/16, sq_2*1j/16,
                1/16, 1/16, -1j/16, 1j/16, sq_2/16, sq_2/16, -sq_2*1j/16, sq_2*1j/16,
                1j/16, 1j/16, 1/16, -1/16, sq_2*1j/16, sq_2*1j/16, sq_2/16, -sq_2/16,
                -1j/16, -1j/16, -1/16, 1/16, -sq_2*1j/16, -sq_2*1j/16, -sq_2/16, sq_2/16,
                sq_2/16, sq_2/16, -sq_2*1j/16, sq_2*1j/16, 2/16, 2/16, -2j/16, 2j/16,
                sq_2/16, sq_2/16, -sq_2*1j/16, sq_2*1j/16, 2/16, 2/16, -2j/16, 2j/16,
                sq_2*1j/16, sq_2*1j/16, sq_2/16, -sq_2/16, 2j/16, 2j/16, 2/16, -2/16,
                -sq_2*1j/16, -sq_2*1j/16, -sq_2/16, sq_2/16, -2j/16, -2j/16, -2/16, 2/16])

    bos_op9 = BosonicOp({"+_0 +_0": 1})
    # Using: truncation = 1
    ref_qubit_op9_tr1 = SparsePauliOp(["II"], coeffs=[0.])
    # Using: truncation = 2
    ref_qubit_op9_tr2 = SparsePauliOp(["XIX", "YIX", "YIY", "XIY"],
                                      coeffs=[sq_2/4, sq_2*1j/4, sq_2/4, -sq_2*1j/4])

    # Test truncation = 1
    @data(
        (bos_op1, ref_qubit_op1_tr1),
        (bos_op2, ref_qubit_op2_tr1),
        (bos_op3, ref_qubit_op3_tr1),
        (bos_op4, ref_qubit_op4_tr1),
        (bos_op5, ref_qubit_op5_tr1),
        (bos_op6, ref_qubit_op6_tr1),
        (bos_op7, ref_qubit_op7_8_tr1),
        (bos_op8, ref_qubit_op7_8_tr1),
        (bos_op9, ref_qubit_op9_tr1),
    )
    @unpack
    def test_mapping_truncation_1(self, bos_op, ref_qubit_op):
        """Test mapping to qubit operator"""
        mapper = BosonicLinearMapper(truncation=1)
        aux = settings.use_pauli_sum_op
        try:
            settings.use_pauli_sum_op = True
            qubit_op = mapper.map(bos_op)
            self.assertEqual(qubit_op, PauliSumOp(ref_qubit_op))
            settings.use_pauli_sum_op = False
            qubit_op = mapper.map(bos_op)
            self.assertEqualSparsePauliOp(qubit_op, ref_qubit_op)
        finally:
            settings.use_pauli_sum_op = aux

    @data(
        (bos_op1, ref_qubit_op1_tr2),
        (bos_op2, ref_qubit_op2_tr2),
        (bos_op3, ref_qubit_op3_tr2),
        (bos_op4, ref_qubit_op4_tr2),
        (bos_op5, ref_qubit_op5_tr2),
        # (bos_op6, ref_qubit_op6_tr2),
        (bos_op7, ref_qubit_op7_8_tr2),
        (bos_op8, ref_qubit_op7_8_tr2),
        (bos_op9, ref_qubit_op9_tr2)
    )
    @unpack
    def test_mapping_truncation_2(self, bos_op, ref_qubit_op):
        """Test mapping to qubit operator"""
        mapper = BosonicLinearMapper(truncation=2)
        aux = settings.use_pauli_sum_op
        try:
            settings.use_pauli_sum_op = True
            qubit_op = mapper.map(bos_op)
            self.assertEqual(qubit_op, PauliSumOp(ref_qubit_op))
            settings.use_pauli_sum_op = False
            qubit_op = mapper.map(bos_op)
            self.assertEqualSparsePauliOp(qubit_op, ref_qubit_op)
        finally:
            settings.use_pauli_sum_op = aux

if __name__ == "__main__":
    unittest.main()
