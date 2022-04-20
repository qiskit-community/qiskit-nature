# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Logarithmic Mapper """

import unittest

from test import QiskitNatureTestCase

from ddt import ddt, data, unpack

from qiskit.opflow import X, Y, Z, I
from qiskit_nature.operators.second_quantization import SpinOp
from qiskit_nature.mappers.second_quantization import LogarithmicMapper, EmbedLocation


@ddt
class TestLogarithmicMapper(QiskitNatureTestCase):
    """Test Logarithmic Mapper"""

    spin_op1 = SpinOp([("Y_0", -0.432 + 1.32j)], 0.5, 1)
    ref_qubit_op1 = (-0.216 + 0.66j) * (Y)

    spin_op2 = SpinOp([("X_0 Z_0 I_1", -1.139 + 0.083j)], 0.5, 2)
    ref_qubit_op2 = (0.02075 + 0.28475j) * (I ^ Y)

    spin_op3 = SpinOp([("X_0 Y_0^2 Z_0 X_1 Y_1 Y_2 Z_2", -0.18 + 1.204j)], 0.5, 3)
    ref_qubit_op3 = (-0.004703125 - 0.000703125j) * (X ^ Z ^ Y)

    spin_op4 = SpinOp([("I_0 Z_1", -0.875 - 0.075j)], 1.5, 2)
    ref_qubit_op4 = (-0.4375 - 0.0375j) * (I ^ Z ^ I ^ I) + (-0.875 - 0.075j) * (Z ^ I ^ I ^ I)

    spin_op5 = SpinOp([("X_0 I_1 I_2 I_3 I_4 I_5 I_6 I_7", 4 + 0j)], 0.5, 8) + SpinOp(
        [("I_0^2 I_2 I_3 I_4 I_5 I_6 I_7 I_8", 8 + 0j)], 0.5, 8
    )

    ref_qubit_op5 = (8.0 + 0j) * (I ^ I ^ I ^ I ^ I ^ I ^ I ^ I) + (2.0 + 0j) * (
        I ^ I ^ I ^ I ^ I ^ I ^ I ^ X
    )

    spin_op6 = SpinOp([("Z_0", -0.875 - 0.075j)], 1, 1)
    ref_qubit_op6 = (
        (-0.4375 - 0.0375j) * (I ^ I) + (0.4375 + 0.0375j) * (I ^ Z) + (-0.875 - 0.075j) * (Z ^ Z)
    )

    ref_qubit_op7 = (
        (-0.4375 - 0.0375j) * (I ^ I) + (-0.4375 - 0.0375j) * (I ^ Z) + (-0.875 - 0.075j) * (Z ^ I)
    )

    @data(
        (spin_op1, ref_qubit_op1),
        (spin_op2, ref_qubit_op2),
        (spin_op3, ref_qubit_op3),
        (spin_op4, ref_qubit_op4),
        (spin_op5, ref_qubit_op5),
        (spin_op6, ref_qubit_op6, 2),
        (spin_op6, ref_qubit_op7, 2, EmbedLocation.LOWER),
    )
    @unpack
    def test_mapping(self, spin_op, ref_qubit_op, padding=1, location=EmbedLocation.UPPER):
        """Test mapping to qubit operator"""
        mapper = LogarithmicMapper(padding, location)
        qubit_op = mapper.map(spin_op)
        self.assertEqual(qubit_op, ref_qubit_op)


if __name__ == "__main__":
    unittest.main()
