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
""" Test VQE UVCC MinimumEigensovler Factory """

import unittest

from test import QiskitNatureTestCase
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.opflow import AerPauliExpectation
from qiskit.algorithms.optimizers import COBYLA
from qiskit_nature.circuit.library.ansatzes import UVCCSD
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import \
    VQEUVCCFactory


class TestVQEUVCCFactory(QiskitNatureTestCase):
    """ Test VQE UVCC MinimumEigensovler Factory """

    # NOTE: The actual usage of this class is mostly tested in combination with the ground-state
    # eigensolvers (one module above).

    def setUp(self):
        super().setUp()

        self.converter = QubitConverter(JordanWignerMapper())

        self.seed = 50
        self.quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                shots=1,
                                                seed_simulator=self.seed,
                                                seed_transpiler=self.seed)

        self._vqe_uvcc_factory = VQEUVCCFactory(self.quantum_instance)

    def test_setters_getters(self):
        """ Test Getter/Setter """

        with self.subTest("Quantum Instance"):
            self.assertEqual(self._vqe_uvcc_factory.quantum_instance, self.quantum_instance)
            self._vqe_uvcc_factory.quantum_instance = None
            self.assertEqual(self._vqe_uvcc_factory.quantum_instance, None)

        with self.subTest("Optimizer"):
            self.assertEqual(self._vqe_uvcc_factory.optimizer, None)
            optimizer = COBYLA()
            self._vqe_uvcc_factory.optimizer = optimizer
            self.assertEqual(self._vqe_uvcc_factory.optimizer, optimizer)

        with self.subTest("Initial Point"):
            self.assertEqual(self._vqe_uvcc_factory.initial_point, None)
            initial_point = [1, 2, 3]
            self._vqe_uvcc_factory.initial_point = initial_point
            self.assertEqual(self._vqe_uvcc_factory.initial_point, initial_point)

        with self.subTest("Expectation"):
            self.assertEqual(self._vqe_uvcc_factory.expectation, None)
            expectation = AerPauliExpectation()
            self._vqe_uvcc_factory.expectation = expectation
            self.assertEqual(self._vqe_uvcc_factory.expectation, expectation)

        with self.subTest("Include Custom"):
            self.assertEqual(self._vqe_uvcc_factory.include_custom, False)
            self._vqe_uvcc_factory.include_custom = True
            self.assertEqual(self._vqe_uvcc_factory.include_custom, True)

        with self.subTest("Variational Form"):
            self.assertEqual(self._vqe_uvcc_factory.ansatz, None)
            ansatz = UVCCSD()
            self._vqe_uvcc_factory.ansatz = ansatz
            self.assertTrue(isinstance(self._vqe_uvcc_factory.ansatz, UVCCSD))

        with self.subTest("Initial State"):
            self.assertEqual(self._vqe_uvcc_factory.initial_state, None)
            initial_state = HartreeFock(4, (1, 1), self.converter)
            self._vqe_uvcc_factory.initial_state = initial_state
            self.assertEqual(self._vqe_uvcc_factory.initial_state, initial_state)


if __name__ == '__main__':
    unittest.main()
