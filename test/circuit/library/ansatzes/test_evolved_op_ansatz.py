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

"""Test the evolved operator ansatz."""

from test import QiskitNatureTestCase
from ddt import ddt, data

from qiskit.opflow import X, H, Z

from qiskit_nature.circuit.library.ansatzes import EvolvedOperatorAnsatz


@ddt
class TestEvolvedOperatorAnsatz(QiskitNatureTestCase):
    """TODO"""

    @data(1, 3)
    def test_evolved_op_ansatz(self, num_qubits):
        """TODO"""

        ops = [Z ^ num_qubits, H ^ num_qubits, X ^ num_qubits]

        evo = EvolvedOperatorAnsatz(ops, insert_barriers=True)
        # print(evo.draw())

        self.assertEqual(evo.num_qubits, num_qubits)
