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

"""Test the UCC Ansatz."""

from test import QiskitNatureTestCase

from ddt import ddt, data, unpack

from qiskit_nature.circuit.library.ansatzes import UCC
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter


def assert_ucc_like_ansatz(test_case, ansatz, num_spin_orbitals, expected_ops):
    """Assertion utility."""
    excitation_ops = ansatz.excitation_ops()

    test_case.assertEqual(len(excitation_ops), len(expected_ops))
    for op, exp in zip(excitation_ops, expected_ops):
        test_case.assertEqual(op._labels, exp._labels)
        test_case.assertEqual(op._coeffs.tolist(), exp._coeffs.tolist())

    ansatz._build()
    test_case.assertEqual(ansatz.num_qubits, num_spin_orbitals)


@ddt
class TestUCC(QiskitNatureTestCase):
    """Tests for the UCC Ansatz."""

    # Note: many variations of this class are tested by its sub-classes UCCSD, PUCCD and SUCCD.
    # Thus, the tests here mainly cover edge cases which those classes cannot account for.

    @unpack
    @data(
        ('t', 8, (2, 2), [FermionicOp([('++--+I-I', 1j), ('--++-I+I', 1j)]),
                          FermionicOp([('++--+II-', 1j), ('--++-II+', 1j)]),
                          FermionicOp([('++--I+-I', 1j), ('--++I-+I', 1j)]),
                          FermionicOp([('++--I+I-', 1j), ('--++I-I+', 1j)]),
                          FermionicOp([('+I-I++--', 1j), ('-I+I--++', 1j)]),
                          FermionicOp([('+II-++--', 1j), ('-II+--++', 1j)]),
                          FermionicOp([('I+-I++--', 1j), ('I-+I--++', 1j)]),
                          FermionicOp([('I+I-++--', 1j), ('I-I+--++', 1j)])]),
        ('t', 8, (2, 1), [FermionicOp([('++--+-II', 1j), ('--++-+II', 1j)]),
                          FermionicOp([('++--+I-I', 1j), ('--++-I+I', 1j)]),
                          FermionicOp([('++--+II-', 1j), ('--++-II+', 1j)])]),
        ('q', 8, (2, 2), [FermionicOp([('++--++--', 1j), ('--++--++', -1j)])]),
        # TODO: add more edge cases?
    )
    def test_ucc_ansatz(self, excitations, num_spin_orbitals, num_particles, expect):
        """Tests the UCC Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = UCC(qubit_converter=converter,
                     num_particles=num_particles,
                     num_spin_orbitals=num_spin_orbitals,
                     excitations=excitations)

        assert_ucc_like_ansatz(self, ansatz, num_spin_orbitals, expect)
