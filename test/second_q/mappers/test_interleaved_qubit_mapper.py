# This code is part of a Qiskit project.
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
        interleaved_mapper = InterleavedQubitMapper(JordanWignerMapper())

        with self.subTest("1-body excitation"):
            ferm_op = FermionicOp({"+_0 -_1": 1}, num_spin_orbitals=4)

            qubit_op = interleaved_mapper.map(ferm_op)

            self.assertEqual(
                qubit_op,
                SparsePauliOp.from_list(
                    [("IXZY", -0.25j), ("IYZY", 0.25), ("IXZX", 0.25), ("IYZX", 0.25j)]
                ),
            )

        with self.subTest("paired 2-body excitation"):
            # NOTE: this is the particularly important test case because we want to observe *NO*
            # Z terms being included in the resulting qubit operator (because they cancel)
            ferm_op = FermionicOp(
                {"+_0 +_4 -_2 -_6": 1, "+_6 +_2 -_4 -_0": -1}, num_spin_orbitals=8
            )

            qubit_op = interleaved_mapper.map(ferm_op)

            self.assertEqual(
                qubit_op,
                SparsePauliOp.from_list(
                    [
                        ("IIYYIIXY", -0.125j),
                        ("IIXXIIXY", 0.125j),
                        ("IIXYIIYY", 0.125j),
                        ("IIYXIIYY", 0.125j),
                        ("IIXYIIXX", -0.125j),
                        ("IIYXIIXX", -0.125j),
                        ("IIYYIIYX", -0.125j),
                        ("IIXXIIYX", 0.125j),
                    ],
                ),
            )


if __name__ == "__main__":
    unittest.main()
