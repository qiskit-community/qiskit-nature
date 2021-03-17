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

from ddt import ddt, data

from qiskit_nature.circuit.library.ansatzes import UCC
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter


@ddt
class TestUCC(QiskitNatureTestCase):
    """Tests for the UCC Ansatz."""

    @data('sd', [1, 2])
    def test_ucc_ansatz(self, excitations):
        """Tests the UCC Ansatz."""
        converter = QubitConverter(JordanWignerMapper())

        ansatz = UCC(qubit_converter=converter, num_particles=[1, 1], num_spin_orbitals=4,
                     excitations=excitations)

        ansatz._build()
        print(vars(ansatz))

        assert ansatz.num_qubits == 4
