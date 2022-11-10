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

""" Test Linear Mapper """

import unittest

from test import QiskitNatureTestCase

from ddt import ddt, data, unpack

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import SpinOp
from qiskit_nature.second_q.mappers import LinearMapper


@ddt
class TestLinearMapper(QiskitNatureTestCase):
    """Test Linear Mapper"""

    spin_op1 = SpinOp({"Y_0^2": -0.432 + 1.32j}, 0.5, 1)
    ref_qubit_op1 = SparsePauliOp(["II", "ZZ"], coeffs=[(-0.054 + 0.165j), (0.054 - 0.165j)])

    spin_op2 = SpinOp({"X_0 Z_0": -1.139 + 0.083j}, 0.5, 2)
    ref_qubit_op2 = SparsePauliOp(
        ["IIYX", "IIXY"], coeffs=[(0.010375 + 0.142375j), (-0.010375 - 0.142375j)]
    )

    spin_op3 = SpinOp({"X_0 Y_0^2 Z_0 X_1 Y_1 Y_2 Z_2": -0.18 + 1.204j}, 0.5, 3)
    ref_qubit_op3 = SparsePauliOp.from_list(
        [
            ("YYIZYX", 0.000587890625 + 8.7890625e-05j),
            ("XXIZYX", 0.000587890625 + 8.7890625e-05j),
            ("YYZIYX", -0.000587890625 - 8.7890625e-05j),
            ("XXZIYX", -0.000587890625 - 8.7890625e-05j),
            ("YYIZXY", -0.000587890625 - 8.7890625e-05j),
            ("XXIZXY", -0.000587890625 - 8.7890625e-05j),
            ("YYZIXY", 0.000587890625 + 8.7890625e-05j),
            ("XXZIXY", 0.000587890625 + 8.7890625e-05j),
        ]
    )

    spin_op4 = SpinOp({"Z_1": -0.875 - 0.075j}, 1.5, 2)
    ref_qubit_op4 = SparsePauliOp.from_list(
        [
            ("ZIIIIIII", -0.65625 - 0.05625j),
            ("IZIIIIII", -0.21875 - 0.01875j),
            ("IIZIIIII", 0.21875 + 0.01875j),
            ("IIIZIIII", 0.65625 + 0.05625j),
        ]
    )

    spin_op5 = SpinOp({"X_0": 4 + 0j}, 0.5, 8) + SpinOp({"": 8 + 0j}, 0.5, 8)
    ref_qubit_op5 = SparsePauliOp.from_list(
        [
            ("IIIIIIIIIIIIIIII", 8.0 + 0j),
            ("IIIIIIIIIIIIIIXX", 1.0 + 0j),
            ("IIIIIIIIIIIIIIYY", 1.0 + 0j),
        ]
    )

    @data(
        (spin_op1, ref_qubit_op1),
        (spin_op2, ref_qubit_op2),
        (spin_op3, ref_qubit_op3),
        (spin_op4, ref_qubit_op4),
        (spin_op5, ref_qubit_op5),
    )
    @unpack
    def test_mapping(self, spin_op, ref_qubit_op):
        """Test mapping to qubit operator"""
        mapper = LinearMapper()
        qubit_op = mapper.map(spin_op)
        self.assertEqual(qubit_op, PauliSumOp(ref_qubit_op))


if __name__ == "__main__":
    unittest.main()
