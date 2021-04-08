# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Hartree Fock initial state circuit."""

import unittest
from test import QiskitNatureTestCase
import numpy as np

from qiskit import QuantumCircuit
from qiskit.opflow.primitive_ops.tapered_pauli_sum_op import Z2Symmetries
from qiskit.quantum_info.operators.symplectic import Pauli
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.circuit.library.initial_states.hartree_fock import hartree_fock_bitstring
from qiskit_nature.mappers.second_quantization import (
    BravyiKitaevMapper, JordanWignerMapper, ParityMapper)
from qiskit_nature.converters.second_quantization import QubitConverter


class TestHartreeFock(QiskitNatureTestCase):
    """ Initial State HartreeFock tests """

    def test_bitstring(self):
        """Simple test for the bitstring function."""
        bitstr = hartree_fock_bitstring(4, (1, 1))
        self.assertTrue(all(bitstr == np.array([True, False, True, False])))

    def test_bitstring_invalid_input(self):
        """Test passing invalid input raises."""

        with self.subTest('too many particles'):
            with self.assertRaises(ValueError):
                _ = hartree_fock_bitstring(4, (3, 3))

        with self.subTest('too few orbitals'):
            with self.assertRaises(ValueError):
                _ = hartree_fock_bitstring(-1, (2, 2))

    def test_qubits_4_jw_h2(self):
        """ qubits 4 jw h2 test """
        state = HartreeFock(4, (1, 1), QubitConverter(JordanWignerMapper()))
        ref = QuantumCircuit(4)
        ref.x([0, 2])
        self.assertEqual(state, ref)

    def test_qubits_4_py_h2(self):
        """ qubits 4 py h2 test """
        state = HartreeFock(4, (1, 1), QubitConverter(ParityMapper()))
        ref = QuantumCircuit(4)
        ref.x([0, 1])
        self.assertEqual(state, ref)

    def test_qubits_4_bk_h2(self):
        """ qubits 4 bk h2 test """
        state = HartreeFock(4, (1, 1), QubitConverter(BravyiKitaevMapper()))
        ref = QuantumCircuit(4)
        ref.x([0, 1, 2])
        self.assertEqual(state, ref)

    def test_qubits_2_py_h2(self):
        """ qubits 2 py h2 test """
        num_particles = (1, 1)
        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        converter.force_match(num_particles=num_particles)
        state = HartreeFock(4, num_particles, converter)
        ref = QuantumCircuit(2)
        ref.x(0)
        self.assertEqual(state, ref)

    def test_qubits_6_py_lih(self):
        """ qubits 6 py lih test """
        num_particles = (1, 1)
        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        z2symmetries = Z2Symmetries(
            symmetries=[Pauli('ZIZIZIZI'), Pauli('ZZIIZZII')],
            sq_paulis=[Pauli('IIIIIIXI'), Pauli('IIIIIXII')],
            sq_list=[2, 3],
            tapering_values=[1, 1],
        )
        converter.force_match(num_particles=num_particles, z2symmetries=z2symmetries)
        state = HartreeFock(10, num_particles, converter)
        ref = QuantumCircuit(6)
        ref.x([0, 1])
        self.assertEqual(state, ref)


if __name__ == '__main__':
    unittest.main()
