# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
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
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.circuit.library.initial_states.hartree_fock import (
    hartree_fock_bitstring,
    hartree_fock_bitstring_mapped,
)
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
)
from qiskit_nature.second_q.mappers import QubitConverter


class TestHartreeFock(QiskitNatureTestCase):
    """Initial State HartreeFock tests"""

    def test_bitstring(self):
        """Simple test for the bitstring function."""
        bitstr = hartree_fock_bitstring(4, (1, 1))
        self.assertTrue(all(bitstr == np.array([True, False, True, False])))

    def test_bitstring_invalid_input(self):
        """Test passing invalid input raises."""

        with self.subTest("too many particles"):
            with self.assertRaises(ValueError):
                _ = hartree_fock_bitstring(4, (3, 3))

        with self.subTest("too few orbitals"):
            with self.assertRaises(ValueError):
                _ = hartree_fock_bitstring(-1, (2, 2))

    def test_qubits_4_jw_h2(self):
        """qubits 4 jw h2 test"""
        state = HartreeFock(4, (1, 1), QubitConverter(JordanWignerMapper()))
        ref = QuantumCircuit(4)
        ref.x([0, 2])
        self.assertEqual(state, ref)

    def test_qubits_4_py_h2(self):
        """qubits 4 py h2 test"""
        state = HartreeFock(4, (1, 1), QubitConverter(ParityMapper()))
        ref = QuantumCircuit(4)
        ref.x([0, 1])
        self.assertEqual(state, ref)

    def test_qubits_4_bk_h2(self):
        """qubits 4 bk h2 test"""
        state = HartreeFock(4, (1, 1), QubitConverter(BravyiKitaevMapper()))
        ref = QuantumCircuit(4)
        ref.x([0, 1, 2])
        self.assertEqual(state, ref)

    def test_qubits_2_py_h2(self):
        """qubits 2 py h2 test"""
        num_particles = (1, 1)
        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        converter.force_match(num_particles=num_particles)
        state = HartreeFock(4, num_particles, converter)
        ref = QuantumCircuit(2)
        ref.x(0)
        self.assertEqual(state, ref)

    def test_qubits_6_py_lih(self):
        """qubits 6 py lih test"""
        num_particles = (1, 1)
        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        z2symmetries = Z2Symmetries(
            symmetries=[Pauli("ZIZIZIZI"), Pauli("ZZIIZZII")],
            sq_paulis=[Pauli("IIIIIIXI"), Pauli("IIIIIXII")],
            sq_list=[2, 3],
            tapering_values=[1, 1],
        )
        converter.force_match(num_particles=num_particles, z2symmetries=z2symmetries)
        state = HartreeFock(10, num_particles, converter)
        ref = QuantumCircuit(6)
        ref.x([0, 1])
        self.assertEqual(state, ref)

    def test_hf_bitstring_mapped(self):
        """Mapped bitstring test for water"""
        # Original driver config when creating operator that resulted in symmetries coded
        # below. The sector [1, -1] is the correct ground sector.
        # PySCFDriver(
        #    atom="O 0.0000 0.0000 0.1173; H 0.0000 0.07572 -0.4692;H 0.0000 -0.07572 -0.4692",
        #    unit=UnitsType.ANGSTROM,
        #    charge=0,
        #    spin=0,
        #    basis='sto-3g',
        #    hf_method=HFMethodType.RHF)
        num_spin_orbitals = 14
        num_particles = (5, 5)
        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        z2symmetries = Z2Symmetries(
            symmetries=[Pauli("IZZIIIIZZIII"), Pauli("ZZIZIIZZIZII")],
            sq_paulis=[Pauli("IIIIIIIIXIII"), Pauli("IIIIIIIIIXII")],
            sq_list=[3, 2],
            tapering_values=[1, -1],
        )
        with self.subTest("Matched bitsring creation"):
            converter.force_match(num_particles=num_particles, z2symmetries=z2symmetries)
            bitstr = hartree_fock_bitstring_mapped(
                num_spin_orbitals=num_spin_orbitals,
                num_particles=num_particles,
                qubit_converter=converter,
            )
            ref_matched = [True, False, True, True, False, True, False, True, False, False]
            self.assertListEqual(bitstr, ref_matched)
        with self.subTest("Bitsring creation with no tapering"):
            bitstr = hartree_fock_bitstring_mapped(
                num_spin_orbitals=num_spin_orbitals,
                num_particles=num_particles,
                qubit_converter=converter,
                match_convert=False,
            )
            ref_notaper = [
                True,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
                False,
                True,
                False,
                False,
            ]
            self.assertListEqual(bitstr, ref_notaper)


if __name__ == "__main__":
    unittest.main()
