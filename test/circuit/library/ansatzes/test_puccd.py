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

"""Test the PUCC Ansatz."""

from test import QiskitNatureTestCase

from ddt import ddt, data, unpack

from qiskit_nature import QiskitNatureError
from qiskit_nature.circuit.library.ansatzes import PUCCD
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter


@ddt
class TestPUCC(QiskitNatureTestCase):
    """Tests for the PUCCD Ansatz."""

    @unpack
    @data(
        (4, (1, 1), [FermionicOp([('+-+-', 1j), ('-+-+', -1j)])]),
        (8, (2, 2), [FermionicOp([('+I-I+I-I', 1j), ('-I+I-I+I', -1j)]),
                     FermionicOp([('+II-+II-', 1j), ('-II+-II+', -1j)]),
                     FermionicOp([('I+-II+-I', 1j), ('I-+II-+I', -1j)]),
                     FermionicOp([('I+I-I+I-', 1j), ('I-I+I-I+', -1j)])]),
    )
    def test_puccd_ansatz(self, num_spin_orbitals, num_particles, expect):
        """Tests the PUCCD Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = PUCCD(qubit_converter=converter,
                       num_particles=num_particles,
                       num_spin_orbitals=num_spin_orbitals)

        ansatz._build()

        self.assertEqual(ansatz.num_qubits, num_spin_orbitals)
        self.assertEqual(len(ansatz.excitation_ops()), len(expect))
        for op, exp in zip(ansatz.excitation_ops(), expect):
            self.assertEqual(op._labels, exp._labels)
            self.assertEqual(op._coeffs, exp._coeffs)

        # TODO: assert actual QuantumCircuit

    @unpack
    @data(
        (4, (1, 1), (True, True), [FermionicOp([('+-II', 1j), ('-+II', 1j)]),
                                   FermionicOp([('II+-', 1j), ('II-+', 1j)]),
                                   FermionicOp([('+-+-', 1j), ('-+-+', -1j)])]),
        (4, (1, 1), (True, False), [FermionicOp([('+-II', 1j), ('-+II', 1j)]),
                                    FermionicOp([('+-+-', 1j), ('-+-+', -1j)])]),
        (4, (1, 1), (False, True), [FermionicOp([('II+-', 1j), ('II-+', 1j)]),
                                    FermionicOp([('+-+-', 1j), ('-+-+', -1j)])]),
    )
    def test_puccd_ansatz_with_singles(self, num_spin_orbitals, num_particles, include_singles,
                                       expect):
        """Tests the PUCCD Ansatz with included single excitations."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = PUCCD(qubit_converter=converter,
                       num_particles=num_particles,
                       num_spin_orbitals=num_spin_orbitals,
                       include_singles=include_singles)

        ansatz._build()

        self.assertEqual(ansatz.num_qubits, num_spin_orbitals)
        self.assertEqual(len(ansatz.excitation_ops()), len(expect))
        for op, exp in zip(ansatz.excitation_ops(), expect):
            self.assertEqual(op._labels, exp._labels)
            self.assertEqual(op._coeffs, exp._coeffs)

        # TODO: assert actual QuantumCircuit

    def test_raise_non_singlet(self):
        """Test an error is raised when the number of alpha and beta electrons differ."""
        with self.assertRaises(QiskitNatureError):
            PUCCD(num_particles=(2, 1))
