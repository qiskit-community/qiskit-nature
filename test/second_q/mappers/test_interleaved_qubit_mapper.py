# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test InterleavedQubitMapper """

import unittest
from test import QiskitNatureTestCase

from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.mappers import InterleavedQubitMapper, JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp


class TestInterleavedQubitMapper(QiskitNatureTestCase):
    """Test InterleavedQubitMapper"""

    def test_mapping(self) -> None:
        """Test the actual mapping procedure."""
        ferm_op = FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=4)

        interleaved_mapper = InterleavedQubitMapper(JordanWignerMapper())

        qubit_op = interleaved_mapper.map(ferm_op).primitive

        self.assertEqual(
            qubit_op,
            SparsePauliOp.from_list(
                [("IXIY", -0.25j), ("IYIY", 0.25), ("IXIX", 0.25), ("IYIX", 0.25j)]
            ),
        )


if __name__ == "__main__":
    unittest.main()
