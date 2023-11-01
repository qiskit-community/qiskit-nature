# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
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

from qiskit import QuantumCircuit
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries
from qiskit.quantum_info.operators.symplectic import Pauli
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.circuit.library.initial_states.hartree_fock import (
    hartree_fock_bitstring_mapped,
)
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevSuperFastMapper,
)
from qiskit_nature.second_q.mappers import TaperedQubitMapper


class TestHartreeFock(QiskitNatureTestCase):
    """Initial State HartreeFock tests"""

    def test_raises_on_unsupported_tapered_mapper(self):
        """Test if an error is raised for an unsupported mapper."""
        with self.assertRaises(NotImplementedError):
            mapper = TaperedQubitMapper(BravyiKitaevSuperFastMapper())
            state = HartreeFock(num_spatial_orbitals=2, num_particles=(1, 1), qubit_mapper=mapper)
            state.draw()

    def test_raises_on_unsupported_mapper_no_mapper(self):
        """Test if an error is raised for an unsupported mapper."""
        with self.assertRaises(NotImplementedError):
            mapper = BravyiKitaevSuperFastMapper()
            state = HartreeFock(num_spatial_orbitals=2, num_particles=(1, 1), qubit_mapper=mapper)
            state.draw()

    def test_qubits_4_jw_h2(self):
        """qubits 4 jw h2 test"""
        state = HartreeFock(2, (1, 1), TaperedQubitMapper(JordanWignerMapper()))
        ref = QuantumCircuit(4)
        ref.x([0, 2])
        self.assertEqual(state, ref)

    def test_qubits_4_jw_h2_lazy_attribute_setting(self):
        """qubits 4 jw h2 with lazy attribute setting test"""
        state = HartreeFock()
        state.num_spatial_orbitals = 2
        state.num_particles = (1, 1)
        state.qubit_mapper = TaperedQubitMapper(JordanWignerMapper())
        ref = QuantumCircuit(4)
        ref.x([0, 2])
        self.assertEqual(state, ref)

    def test_qubits_4_py_h2(self):
        """qubits 4 py h2 test"""
        state = HartreeFock(2, (1, 1), TaperedQubitMapper(ParityMapper()))
        ref = QuantumCircuit(4)
        ref.x([0, 1])
        self.assertEqual(state, ref)

    def test_qubits_4_bk_h2(self):
        """qubits 4 bk h2 test"""
        state = HartreeFock(2, (1, 1), TaperedQubitMapper(BravyiKitaevMapper()))
        ref = QuantumCircuit(4)
        ref.x([0, 1, 2])
        self.assertEqual(state, ref)

    def test_qubits_2_py_h2(self):
        """qubits 2 py h2 test"""
        num_particles = (1, 1)
        mapper = TaperedQubitMapper(ParityMapper(num_particles))
        state = HartreeFock(2, num_particles, mapper)
        ref = QuantumCircuit(2)
        ref.x(0)
        self.assertEqual(state, ref)

    def test_qubits_6_py_lih(self):
        """qubits 6 py lih test"""
        num_particles = (1, 1)
        z2symmetries = Z2Symmetries(
            symmetries=[Pauli("ZIZIZIZI"), Pauli("ZZIIZZII")],
            sq_paulis=[Pauli("IIIIIIXI"), Pauli("IIIIIXII")],
            sq_list=[2, 3],
            tapering_values=[1, 1],
        )
        mapper = TaperedQubitMapper(
            ParityMapper(num_particles=num_particles), z2symmetries=z2symmetries
        )
        state = HartreeFock(5, num_particles, mapper)
        ref = QuantumCircuit(6)
        ref.x([0, 1])
        self.assertEqual(state, ref)

    def test_hf_bitstring_mapped(self):
        """Mapped bitstring test for water"""

        num_spatial_orbitals = 7
        num_particles = (5, 5)
        z2symmetries = Z2Symmetries(
            symmetries=[Pauli("IZZIIIIZZIII"), Pauli("ZZIZIIZZIZII")],
            sq_paulis=[Pauli("IIIIIIIIXIII"), Pauli("IIIIIIIIIXII")],
            sq_list=[3, 2],
            tapering_values=[1, -1],
        )
        mapper = TaperedQubitMapper(
            ParityMapper(num_particles=num_particles), z2symmetries=z2symmetries
        )

        bitstr = hartree_fock_bitstring_mapped(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
        )

        ref_matched = [True, False, True, True, False, True, False, True, False, False]
        self.assertListEqual(bitstr, ref_matched)


if __name__ == "__main__":
    unittest.main()
