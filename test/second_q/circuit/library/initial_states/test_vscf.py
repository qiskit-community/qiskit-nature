# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the VSCF initial state."""

import unittest
from test import QiskitNatureTestCase
import numpy as np

from qiskit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import VSCF
from qiskit_nature.second_q.circuit.library.initial_states.vscf import vscf_bitstring
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.mappers import QubitConverter


class TestVSCF(QiskitNatureTestCase):
    """Initial State vscf tests"""

    def test_bitstring(self):
        """Test the vscf_bitstring method."""
        bitstr = vscf_bitstring([2, 2])
        self.assertTrue(all(bitstr == np.array([True, False, True, False])))  # big endian

    def test_qubits_4(self):
        """Test 2 modes 2 modals."""
        num_modals = [2, 2]
        vscf = VSCF(num_modals)
        ref = QuantumCircuit(4)
        ref.x([0, 2])

        self.assertEqual(ref, vscf)

    def test_qubits_5(self):
        """Test 2 modes 2 modals for the first mode and 3 modals for the second."""
        num_modals = [2, 3]
        vscf = VSCF(num_modals)
        ref = QuantumCircuit(5)
        ref.x([0, 2])

        self.assertEqual(ref, vscf)

    def test_qubits_6_lazy_attribute_setting(self):
        """Test 2 modes 2 modal for the first mode and 4 modals for the second
        with lazy attribute setting."""
        num_modals = [2, 4]
        qubit_converter = QubitConverter(ParityMapper())
        vscf = VSCF()
        vscf.num_modals = num_modals
        vscf.qubit_converter = qubit_converter
        ref = QuantumCircuit(6)
        ref.x([0, 2])

        self.assertEqual(ref, vscf)


if __name__ == "__main__":
    unittest.main()
