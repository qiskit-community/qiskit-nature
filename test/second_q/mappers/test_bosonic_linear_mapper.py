# This code is part of a Qiskit project.
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

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import BosonicOp
from qiskit_nature.second_q.mappers import BosonicLinearMapper


@ddt
class TestBosonicLinearMapper(QiskitNatureTestCase):
    """Test Bosonic Linear Mapper"""

    # Define some useful coefficients
    sq_2 = np.sqrt(2)

    bos_op1 = BosonicOp({"+_0": 1})
    # Using: max_occupation = 1
    ref_qubit_op1_tr1 = SparsePauliOp(["XX", "YY", "YX", "XY"], coeffs=[0.25, 0.25, -0.25j, 0.25j])
    # Using: max_occupation = 2
    ref_qubit_op1_tr2 = SparsePauliOp(
        ["IXX", "IYY", "IYX", "IXY", "XXI", "YYI", "YXI", "XYI"],
        coeffs=[0.25, 0.25, -0.25j, 0.25j, sq_2 / 4, sq_2 / 4, -1j * sq_2 / 4, 1j * sq_2 / 4],
    )

    bos_op2 = BosonicOp({"-_0": 1})
    # Using: max_occupation = 1
    ref_qubit_op2_tr1 = SparsePauliOp(["XX", "YY", "YX", "XY"], coeffs=[0.25, 0.25, 0.25j, -0.25j])
    # Using: max_occupation = 2
    ref_qubit_op2_tr2 = SparsePauliOp(
        ["IXX", "IYY", "IYX", "IXY", "XXI", "YYI", "YXI", "XYI"],
        coeffs=[0.25, 0.25, 0.25j, -0.25j, sq_2 / 4, sq_2 / 4, 1j * sq_2 / 4, -1j * sq_2 / 4],
    )

    bos_op3 = BosonicOp({"+_1": 1})
    # Using: max_occupation = 1
    ref_qubit_op3_tr1 = SparsePauliOp(
        ["XXII", "YYII", "YXII", "XYII"], coeffs=[0.25, 0.25, -0.25j, 0.25j]
    )
    # Using: max_occupation = 2
    ref_qubit_op3_tr2 = SparsePauliOp(
        ["IXXIII", "IYYIII", "IYXIII", "IXYIII", "XXIIII", "YYIIII", "YXIIII", "XYIIII"],
        coeffs=[0.25, 0.25, -0.25j, 0.25j, sq_2 / 4, sq_2 / 4, -1j * sq_2 / 4, 1j * sq_2 / 4],
    )

    bos_op4 = BosonicOp({"-_1": 1})
    # Using: max_occupation = 1
    ref_qubit_op4_tr1 = SparsePauliOp(
        ["XXII", "YYII", "YXII", "XYII"], coeffs=[0.25, 0.25, 0.25j, -0.25j]
    )
    # Using: max_occupation = 2
    ref_qubit_op4_tr2 = SparsePauliOp(
        ["IXXIII", "IYYIII", "IYXIII", "IXYIII", "XXIIII", "YYIIII", "YXIIII", "XYIIII"],
        coeffs=[0.25, 0.25, 0.25j, -0.25j, sq_2 / 4, sq_2 / 4, 1j * sq_2 / 4, -1j * sq_2 / 4],
    )

    bos_op5 = BosonicOp({"+_0 -_0": 1})
    # Using: max_occupation = 1
    ref_qubit_op5_tr1 = SparsePauliOp(["II", "ZZ", "ZI", "IZ"], coeffs=[0.25, -0.25, -0.25, 0.25])
    # Using: max_occupation = 2
    ref_qubit_op5_tr2 = SparsePauliOp(
        ["III", "IZZ", "IZI", "IIZ", "ZZI", "ZII"], coeffs=[0.75, -0.25, 0.25, 0.25, -0.5, -0.5]
    )

    bos_op6 = BosonicOp({"-_0 +_0": 1})
    # Using: max_occupation = 1
    ref_qubit_op6_tr1 = SparsePauliOp(["II", "ZZ", "ZI", "IZ"], coeffs=[0.25, -0.25, +0.25, -0.25])
    # Using: max_occupation = 2
    ref_qubit_op6_tr2 = SparsePauliOp(
        ["III", "IZZ", "IZI", "IIZ", "ZZI", "ZII"], coeffs=[0.75, -0.25, -0.25, -0.25, -0.5, 0.5]
    )

    bos_op7 = BosonicOp({"+_0 -_1": 1})
    bos_op8 = BosonicOp({"-_1 +_0": 1})
    # Using: max_occupation = 1
    # fmt: off
    ref_qubit_op7_8_tr1 = SparsePauliOp(
        ["XXXX", "XXYY", "XXYX", "XXXY", "YYXX", "YYYY", "YYYX", "YYXY",
         "YXXX", "YXYY", "YXYX", "YXXY", "XYXX", "XYYY", "XYYX", "XYXY",],
        coeffs=[
         1 / 16, 1 / 16, -1j / 16, 1j / 16, 1 / 16, 1 / 16, -1j / 16, 1j / 16,
         1j / 16, 1j / 16, 1 / 16, -1 / 16, -1j / 16, -1j / 16, -1 / 16, 1 / 16,]
    )
    # fmt: on
    # Using: max_occupation = 2
    # fmt: off
    ref_qubit_op7_8_tr2 = SparsePauliOp(
        ["IXXIXX", "IXXIYY", "IXXIYX", "IXXIXY", "IXXXXI", "IXXYYI", "IXXYXI", "IXXXYI",
         "IYYIXX", "IYYIYY", "IYYIYX", "IYYIXY", "IYYXXI", "IYYYYI", "IYYYXI", "IYYXYI",
         "IYXIXX", "IYXIYY", "IYXIYX", "IYXIXY", "IYXXXI", "IYXYYI", "IYXYXI", "IYXXYI",
         "IXYIXX", "IXYIYY", "IXYIYX", "IXYIXY", "IXYXXI", "IXYYYI", "IXYYXI", "IXYXYI",
         "XXIIXX", "XXIIYY", "XXIIYX", "XXIIXY", "XXIXXI", "XXIYYI", "XXIYXI", "XXIXYI",
         "YYIIXX", "YYIIYY", "YYIIYX", "YYIIXY", "YYIXXI", "YYIYYI", "YYIYXI", "YYIXYI",
         "YXIIXX", "YXIIYY", "YXIIYX", "YXIIXY", "YXIXXI", "YXIYYI", "YXIYXI", "YXIXYI",
         "XYIIXX", "XYIIYY", "XYIIYX", "XYIIXY", "XYIXXI", "XYIYYI", "XYIYXI", "XYIXYI",],
        coeffs=[
         1 / 16, 1 / 16, -1j / 16, 1j / 16, sq_2 / 16, sq_2 / 16, -sq_2 * 1j / 16, sq_2 * 1j / 16,
         1 / 16, 1 / 16, -1j / 16, 1j / 16, sq_2 / 16, sq_2 / 16, -sq_2 * 1j / 16, sq_2 * 1j / 16,
         1j / 16, 1j / 16, 1 / 16, -1 / 16, sq_2 * 1j / 16, sq_2 * 1j / 16, sq_2 / 16, -sq_2 / 16,
         -1j / 16, -1j / 16, -1 / 16, 1 / 16, -sq_2 * 1j / 16, -sq_2 * 1j / 16, -sq_2 / 16, sq_2 / 16,
         sq_2 / 16, sq_2 / 16, -sq_2 * 1j / 16, sq_2 * 1j / 16, 2 / 16, 2 / 16, -2j / 16, 2j / 16,
         sq_2 / 16, sq_2 / 16, -sq_2 * 1j / 16, sq_2 * 1j / 16, 2 / 16, 2 / 16, -2j / 16, 2j / 16,
         sq_2 * 1j / 16, sq_2 * 1j / 16, sq_2 / 16, -sq_2 / 16, 2j / 16, 2j / 16, 2 / 16, -2 / 16,
         -sq_2 * 1j / 16, -sq_2 * 1j / 16, -sq_2 / 16, sq_2 / 16, -2j / 16, -2j / 16, -2 / 16, 2 / 16,]
    )
    # fmt: on

    bos_op9 = BosonicOp({"+_0 +_0": 1})
    # Using: max_occupation = 1
    ref_qubit_op9_tr1 = SparsePauliOp(["II"], coeffs=[0.0])
    # Using: max_occupation = 2
    ref_qubit_op9_tr2 = SparsePauliOp(
        ["XIX", "YIX", "YIY", "XIY"], coeffs=[sq_2 / 4, -sq_2 * 1j / 4, sq_2 / 4, sq_2 * 1j / 4]
    )

    # Test max_occupation = 1
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
    def test_mapping_max_occupation_1(self, bos_op, ref_qubit_op):
        """Test mapping to qubit operator"""
        mapper = BosonicLinearMapper(max_occupation=1)
        qubit_op = mapper.map(bos_op)
        self.assertEqualSparsePauliOp(qubit_op, ref_qubit_op)

    @data(
        (bos_op1, ref_qubit_op1_tr2),
        (bos_op2, ref_qubit_op2_tr2),
        (bos_op3, ref_qubit_op3_tr2),
        (bos_op4, ref_qubit_op4_tr2),
        (bos_op5, ref_qubit_op5_tr2),
        (bos_op6, ref_qubit_op6_tr2),
        (bos_op7, ref_qubit_op7_8_tr2),
        (bos_op8, ref_qubit_op7_8_tr2),
        (bos_op9, ref_qubit_op9_tr2),
    )
    @unpack
    def test_mapping_max_occupation_2(self, bos_op, ref_qubit_op):
        """Test mapping to qubit operator"""
        mapper = BosonicLinearMapper(max_occupation=2)
        qubit_op = mapper.map(bos_op)
        self.assertEqualSparsePauliOp(qubit_op, ref_qubit_op)


if __name__ == "__main__":
    unittest.main()
