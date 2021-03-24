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

"""Test the UVCC Ansatz."""

from test import QiskitNatureTestCase

from ddt import ddt, data, unpack

from qiskit_nature.circuit.library.ansatzes import UVCC
from qiskit_nature.mappers.second_quantization import DirectMapper
from qiskit_nature.operators.second_quantization import VibrationalOp
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter


def assert_ucc_like_ansatz(test_case, ansatz, num_modals, expected_ops):
    """Assertion utility."""
    excitation_ops = ansatz.excitation_ops()

    test_case.assertEqual(len(excitation_ops), len(expected_ops))
    for op, exp in zip(excitation_ops, expected_ops):
        test_case.assertEqual(op._labels, exp._labels)
        test_case.assertEqual(op._coeffs.tolist(), exp._coeffs.tolist())

    ansatz._build()
    test_case.assertEqual(ansatz.num_qubits, sum(num_modals))


@ddt
class TestUVCC(QiskitNatureTestCase):
    """Tests for the UVCC Ansatz."""

    @unpack
    @data(
        ('s', [2], [VibrationalOp([('+-', 1j), ('-+', 1j)], 1, 2)]),
        ('s', [2, 2], [VibrationalOp([('+-II', 1j), ('-+II', 1j)], 2, 2),
                       VibrationalOp([('II+-', 1j), ('II-+', 1j)], 2, 2)]),
    )
    def test_ucc_ansatz(self, excitations, num_modals, expect):
        """Tests the UVCC Ansatz."""
        converter = QubitConverter(DirectMapper())

        ansatz = UVCC(qubit_converter=converter,
                      num_modals=num_modals,
                      excitations=excitations)

        assert_ucc_like_ansatz(self, ansatz, num_modals, expect)
