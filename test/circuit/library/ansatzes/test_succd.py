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

"""Test the SUCCD Ansatz."""

from test import QiskitNatureTestCase

from qiskit_nature.circuit.library.ansatzes import SUCCD
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter


class TestSUCCD(QiskitNatureTestCase):
    """Tests for the UCCSD Ansatz."""

    def test_ucc_ansatz(self):
        """Tests the UCCSD Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = SUCCD(qubit_converter=converter, num_particles=[2, 2], num_spin_orbitals=8)

        ansatz._build()

        assert ansatz.num_qubits == 8
