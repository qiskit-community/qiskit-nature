# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Bosonic Logarithmic Mapper """

import unittest

from test import QiskitNatureTestCase

from ddt import ddt, data, unpack
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import BosonicOp
from qiskit_nature.second_q.mappers import BosonicLinearMapper
from qiskit_nature.second_q.mappers import BosonicLogarithmicMapper


@ddt
class TestBosonicLogarithmicMapper(QiskitNatureTestCase):
    """Test Bosonic Logarithmic Mapper"""

    # Define some useful coefficients
    sq_2 = np.sqrt(2)
    sq_3 = np.sqrt(3)
    sq_5 = np.sqrt(5)
    sq_6 = np.sqrt(6)
    sq_7 = np.sqrt(7)

    bos_op1 = BosonicOp({"+_0": 1})
    # Using: max_occupation = 3 (number_of_qubits_per_mode = 2)
    ref_qubit_op1_tr1 = 0.25 * SparsePauliOp(
        ["IX", "IY", "ZX", "ZY", "XX", "XY", "YX", "YY"],
        coeffs=[1 + sq_3, -1j*(1 + sq_3), 1 - sq_3, -1j*(1 - sq_3), sq_2, 1j*sq_2, -1j*sq_2, sq_2])
    # Using: max_occupation = 7 (number_of_qubits_per_mode = 3)
    ref_qubit_op1_tr2 = 0.125 * SparsePauliOp(
        ["IIX", "IIY", "IZX", "IZY",
         "ZIX", "ZIY", "ZZX", "ZZY",
         "IXX", "IXY", "IYX", "IYY",
         "ZXX", "ZXY", "ZYX", "ZYY",
         "XXX", "XXY", "XYX", "XYY",
         "YXX", "YXY", "YYX", "YYY"],
        coeffs=[1+sq_3+sq_5+sq_7, -1j*(1+sq_3+sq_5+sq_7), 1-sq_3+sq_5-sq_7, -1j*(1-sq_3+sq_5-sq_7),
                1+sq_3-sq_5-sq_7, -1j*(1+sq_3-sq_5-sq_7), 1-sq_3-sq_5+sq_7, -1j*(1-sq_3-sq_5+sq_7),
                sq_2 + sq_6, 1j*(sq_2 + sq_6), -1j*(sq_2 + sq_6), sq_2 + sq_6,
                sq_2 - sq_6, 1j*(sq_2 - sq_6), -1j*(sq_2 - sq_6), sq_2 - sq_6,
                2, 2j, 2j, -2,
                -2j, 2, 2, 2j],
    )

    # Test max_occupation = 7 (number_of_qubits_per_mode = 2)
    @data(
        (bos_op1, ref_qubit_op1_tr1)
    )
    @unpack
    def test_mapping_max_occupation_3(self, bos_op, ref_qubit_op):
        """Test mapping to qubit operator"""
        mapper = BosonicLogarithmicMapper(max_occupation=3)
        qubit_op = mapper.map(bos_op)
        self.assertEqualSparsePauliOp(qubit_op, ref_qubit_op)

    @data(
        (bos_op1, ref_qubit_op1_tr2)
    )
    @unpack
    def test_mapping_max_occupation_7(self, bos_op, ref_qubit_op):
        """Test mapping to qubit operator"""
        mapper = BosonicLogarithmicMapper(max_occupation=7)
        qubit_op = mapper.map(bos_op)
        self.assertEqualSparsePauliOp(qubit_op, ref_qubit_op)


if __name__ == "__main__":
    unittest.main()
