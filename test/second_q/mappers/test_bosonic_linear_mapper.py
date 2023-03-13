# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
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
    """Test Boonic Linear Mapper"""

    # Define some userful coefficients
    sq_2 = np.sqrt(2) / 4
    sq_3 = np.sqrt(3) / 4

    # +_0 -_0, -_0 +_0, +_0 -_1, -_1 +_0, +_0 +_0

    bos_op1 = BosonicOp({"+_0": 1})
    # Using: truncation = 1
    ref_qubit_op1_tr1 = SparsePauliOp(["XX", "YY", "XY", "YX"], coeffs=[0.25, 0.25, -0.25j, 0.25j])
    # Using: truncation = 2
    ref_qubit_op1_tr2 = SparsePauliOp(["IXX", "IYY", "IXY", "IYX", "XXI", "YYI", "XYI", "YXI"],
                                      coeffs=[0.25, 0.25, -0.25j, 0.25j, sq_2, sq_2, -1j*sq_2, 1j*sq_2])

    bos_op2 = BosonicOp({"-_0": 1})
    # Using: truncation = 1
    ref_qubit_op2_tr1 = SparsePauliOp(["XX", "YY", "XY", "YX"], coeffs=[0.25, 0.25, 0.25j, -0.25j])
    # Using: truncation = 2
    ref_qubit_op2_tr2 = SparsePauliOp(["IXX", "IYY", "IXY", "IYX", "XXI", "YYI", "XYI", "YXI"],
                                      coeffs=[0.25, 0.25, 0.25j, -0.25j, sq_2, sq_2, 1j*sq_2, -1j*sq_2])

    bos_op3 = BosonicOp({"+_1": 1})
    # Using: truncation = 1
    ref_qubit_op3_tr1 = SparsePauliOp(["XXII", "YYII", "XYII", "YXII"], 
                                      coeffs=[0.25, 0.25, -0.25j, 0.25j])
    # Using: truncation = 2
    ref_qubit_op3_tr2 = SparsePauliOp(
        ["IXXIII", "IYYIII", "IXYIII", "IYXIII", "XXIIII", "YYIIII", "XYIIII", "YXIIII"],
        coeffs=[0.25, 0.25, -0.25j, 0.25j, sq_2, sq_2, -1j*sq_2, 1j*sq_2]
    )

    bos_op4 = BosonicOp({"-_1": 1})
    # Using: truncation = 1
    ref_qubit_op4_tr1 = SparsePauliOp(["XXII", "YYII", "XYII", "YXII"], 
                                      coeffs=[0.25, 0.25, 0.25j, -0.25j])
    # Using: truncation = 2
    ref_qubit_op4_tr2 = SparsePauliOp(
        ["IXXIII", "IYYIII", "IXYIII", "IYXIII", "XXIIII", "YYIIII", "XYIIII", "YXIIII"],
        coeffs=[0.25, 0.25, 0.25j, -0.25j, sq_2, sq_2, 1j*sq_2, -1j*sq_2]
    )

    bos_op5 = BosonicOp({"+_0 -_0": 1})
    # Using: truncation = 1
    ref_qubit_op5_tr1 = SparsePauliOp(["II", "ZZ", "IZ", "ZI"], coeffs=[0.25, -0.25, -0.25, 0.25])
    # Using: truncation = 2
    ref_qubit_op5_tr2 = SparsePauliOp(["III", "IZZ", "IIZ", "IZI", "III", "ZZI", "IZI", "ZII"],
                                      coeffs=[0.25, -0.25, -0.25, 0.25, sq_2, -sq_2, -sq_2, sq_2])

    # Test truncation = 1
    @data(
        (bos_op1, ref_qubit_op1_tr1),
        (bos_op2, ref_qubit_op2_tr1),
        (bos_op3, ref_qubit_op3_tr1),
        (bos_op4, ref_qubit_op4_tr1),
        (bos_op5, ref_qubit_op5_tr1),
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

    # def test_mapping_overwrite_reg_len(self):
    #     """Test overwriting the register length."""
    #     op = SpinOp({"Y_0^2": -0.432 + 1.32j}, 0.5, 1)
    #     expected = SpinOp({"Y_0^2": -0.432 + 1.32j}, 0.5, 3)
    #     mapper = LinearMapper()
    #     self.assertEqual(mapper.map(op, register_length=3), mapper.map(expected))


if __name__ == "__main__":
    unittest.main()
