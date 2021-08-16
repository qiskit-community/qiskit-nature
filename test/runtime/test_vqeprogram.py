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

"""Test the VQE program."""

from test import QiskitNatureTestCase

import unittest
from ddt import ddt, data
import numpy as np
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.algorithms import VQEResult
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import I, Z

from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.runtime import VQEProgram

from .fake_vqeruntime import FakeRuntimeProvider


@ddt
class TestVQEProgram(QiskitNatureTestCase):
    """Test the VQE program."""

    def setUp(self):
        super().setUp()
        self.provider = FakeRuntimeProvider()

    def get_standard_program(self):
        """Get a standard VQEProgram and operator to find the ground state of."""
        circuit = RealAmplitudes(3)
        operator = Z ^ I ^ Z
        initial_point = np.random.random(circuit.num_parameters)
        backend = QasmSimulatorPy()

        vqe = VQEProgram(
            ansatz=circuit,
            optimizer=SPSA(),
            initial_point=initial_point,
            backend=backend,
            provider=self.provider,
        )
        return vqe, operator

    @data({"name": "SPSA", "maxiter": 100}, SPSA(maxiter=100))
    def test_standard_case(self, optimizer):
        """Test a standard use case."""
        vqe, operator = self.get_standard_program()
        vqe.optimizer = optimizer
        result = vqe.compute_minimum_eigenvalue(operator)

        self.assertIsInstance(result, VQEResult)

    def test_supports_aux_ops(self):
        """Test the VQEProgram says it supports aux operators."""
        vqe, _ = self.get_standard_program()
        self.assertTrue(vqe.supports_aux_operators)

    def test_return_groundstate(self):
        """Test the VQEProgram yields a ground state solver that returns the ground state."""
        vqe, _ = self.get_standard_program()
        qubit_converter = QubitConverter(JordanWignerMapper())
        gss = GroundStateEigensolver(qubit_converter, vqe)
        self.assertTrue(gss.returns_groundstate)


if __name__ == "__main__":
    unittest.main()
